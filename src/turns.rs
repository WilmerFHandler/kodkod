//! Turn boundaries over a message transcript.

use crate::Message;

/// Message indices where each user turn starts.
pub fn user_turn_starts(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| match message {
            Message::User(_) => Some(index),
            _ => None,
        })
        .collect()
}

/// Inclusive start and exclusive end for each user turn.
pub fn turn_ranges(messages: &[Message]) -> Vec<(usize, usize)> {
    let starts = user_turn_starts(messages);
    if starts.is_empty() {
        return Vec::new();
    }
    starts
        .iter()
        .enumerate()
        .map(|(i, &start)| {
            let end = starts
                .get(i + 1)
                .copied()
                .unwrap_or(messages.len());
            (start, end)
        })
        .collect()
}

/// Whether a turn segment contains tool calls or tool results.
pub fn segment_uses_tools(messages: &[Message]) -> bool {
    messages.iter().any(|msg| match msg {
        Message::Assistant(a) => !a.tool_calls().is_empty(),
        Message::ToolResult(_) => true,
        _ => false,
    })
}

/// Last non-empty assistant text in a message slice (typically one turn).
pub fn final_assistant_text(messages: &[Message]) -> Option<&str> {
    messages.iter().rev().find_map(|msg| match msg {
        Message::Assistant(a) => {
            let content = a.content();
            (!content.is_empty()).then_some(content)
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssistantMessage, UserMessage};

    #[test]
    fn turn_ranges_cover_transcript() {
        let messages = vec![
            Message::User(UserMessage::new("a")),
            Message::Assistant(AssistantMessage::new("1")),
            Message::User(UserMessage::new("b")),
        ];
        assert_eq!(turn_ranges(&messages), vec![(0, 2), (2, 3)]);
    }
}