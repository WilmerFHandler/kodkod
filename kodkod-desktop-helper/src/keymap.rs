//! Resolve input against the compositor's keymap, never an assumed layout.
use anyhow::{Context, Result, ensure};
use xkbcommon::xkb;

#[derive(Clone, Copy, Default)]
pub struct Modifiers {
    pub depressed: u32,
    pub latched: u32,
    pub locked: u32,
    pub group: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stroke {
    pub key: u32,
    pub modifiers: Vec<u32>,
}

pub struct KeyboardMap {
    map: xkb::Keymap,
    pub modifiers: Modifiers,
}
impl KeyboardMap {
    pub fn parse(text: String, modifiers: Modifiers) -> Result<Self> {
        ensure!(!text.contains('\0'), "Invalid keyboard map");
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let map = xkb::Keymap::new_from_string(
            &context,
            text,
            xkb::KEYMAP_FORMAT_TEXT_V1,
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .context("Could not read compositor keyboard map")?;
        ensure!(
            map.max_keycode().raw() <= 767,
            "Keyboard map keycodes exceed the evdev range"
        );
        Ok(Self { map, modifiers })
    }
    fn mask(&self, name: &str) -> u32 {
        let index = self.map.mod_get_index(name);
        if index < 32 { 1 << index } else { 0 }
    }
    fn modifier(&self, name: &str) -> Option<(u32, u32)> {
        let key = self.map.key_by_name(name)?;
        let mut state = xkb::State::new(&self.map);
        state.update_key(key, xkb::KeyDirection::Down);
        let mask = state.serialize_mods(xkb::STATE_MODS_DEPRESSED);
        (mask != 0).then_some((key.raw().checked_sub(8)?, mask))
    }
    fn state(&self, extra: u32) -> xkb::State {
        let mut state = xkb::State::new(&self.map);
        let m = self.modifiers;
        state.update_mask(m.depressed | extra, m.latched, m.locked, 0, 0, m.group);
        state
    }
    /// Resolve the entire string before emitting any events, including failures
    /// late in a string. Latched modifiers are consumed per key and cannot be
    /// modeled by one stable batch, so ask for a fresh state instead.
    pub fn text(&self, text: &str) -> Result<Vec<Stroke>> {
        ensure!(
            self.modifiers.latched == 0,
            "Release latched keyboard modifiers before typing"
        );
        ensure!(
            self.modifiers.depressed
                & (self.mask("Control") | self.mask("Mod1") | self.mask("Mod4"))
                == 0,
            "Release Control, Alt and Super before typing"
        );
        let shift = self.modifier("LFSH");
        let altgr = self.modifier("RALT");
        let combinations = [
            vec![],
            shift.into_iter().collect(),
            altgr.into_iter().collect(),
            shift.into_iter().chain(altgr).collect(),
        ];
        text.chars()
            .map(|character| {
                for modifiers in &combinations {
                    let extra = modifiers.iter().fold(0, |mask, (_, bit)| mask | bit);
                    let state = self.state(extra);
                    for raw in self.map.min_keycode().raw()..=self.map.max_keycode().raw() {
                        let key = xkb::Keycode::new(raw);
                        let actual = state.key_get_utf8(key);
                        if actual == character.to_string() || (character == '\n' && actual == "\r")
                        {
                            let key = raw.checked_sub(8).context("Invalid EIS keyboard keycode")?;
                            return Ok(Stroke {
                                key,
                                modifiers: modifiers
                                    .iter()
                                    .filter(|(_, mask)| self.modifiers.depressed & mask != *mask)
                                    .map(|(key, _)| *key)
                                    .collect(),
                            });
                        }
                    }
                }
                anyhow::bail!(
                    "Text is not representable by the current keyboard layout; no text was inserted"
                )
            })
            .collect()
    }
    pub fn chord(&self, key: &str) -> Result<Stroke> {
        let (modifier, letter) = match key.split_once('+') {
            Some(("ctrl", letter)) => ("LCTL", letter),
            Some(("cmd", letter)) => ("LWIN", letter),
            _ => anyhow::bail!("Unsupported keyboard chord"),
        };
        ensure!(
            matches!(letter, "a" | "c" | "v" | "l"),
            "Unsupported keyboard chord"
        );
        ensure!(
            self.modifiers.latched == 0,
            "Release latched modifiers before a shortcut"
        );
        let (modifier_key, modifier_mask) = self
            .modifier(modifier)
            .context("Keyboard modifier unavailable")?;
        ensure!(
            self.modifiers.depressed & !modifier_mask == 0,
            "Release other held modifiers before a shortcut"
        );
        let mut plain = xkb::State::new(&self.map);
        plain.update_mask(0, 0, 0, 0, 0, self.modifiers.group);
        for raw in self.map.min_keycode().raw()..=self.map.max_keycode().raw() {
            if plain.key_get_utf8(xkb::Keycode::new(raw)) == letter {
                return Ok(Stroke {
                    key: raw.checked_sub(8).context("Invalid EIS keycode")?,
                    modifiers: if self.modifiers.depressed & modifier_mask == modifier_mask {
                        vec![]
                    } else {
                        vec![modifier_key]
                    },
                });
            }
        }
        anyhow::bail!("Shortcut is not representable by the current keyboard layout")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn map(layout: &str) -> KeyboardMap {
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let map = xkb::Keymap::new_from_names(
            &context,
            "",
            "",
            layout,
            "",
            Some(String::new()),
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .unwrap();
        KeyboardMap::parse(
            map.get_as_string(xkb::KEYMAP_FORMAT_TEXT_V1),
            Modifiers::default(),
        )
        .unwrap()
    }
    #[test]
    fn text_uses_actual_layout_and_preflights_unmappable_suffix() {
        let us = map("us");
        assert!(us.text("Hello 123!\n").is_ok());
        assert!(us.text("ordinary text 🦊").is_err());
        let se = map("se");
        assert!(se.text("Hej Åäö @€").is_ok());
        assert_ne!(us.text("@").unwrap(), se.text("@").unwrap());
        assert!(se.chord("ctrl+l").is_ok());
    }
    #[test]
    fn caps_lock_is_preserved_and_held_modifiers_are_not_released() {
        let mut keyboard = map("us");
        keyboard.modifiers.locked = keyboard.mask("Lock");
        assert!(keyboard.text("a").unwrap()[0].modifiers.contains(&42));
        assert!(keyboard.text("A").unwrap()[0].modifiers.is_empty());
        keyboard.modifiers.depressed = keyboard.mask("Control");
        assert!(keyboard.text("a").is_err());
        assert!(keyboard.chord("ctrl+a").unwrap().modifiers.is_empty());
        keyboard.modifiers.latched = keyboard.mask("Shift");
        assert!(keyboard.text("a").is_err());
    }
}
