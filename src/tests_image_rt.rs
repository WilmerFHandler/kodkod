use super::*;

#[test]
fn image_in_conversation_json_roundtrips() {
    use lynx_agent::{Conversation, Image};
    let mut conversation = Conversation::new();
    conversation.push_user_message_with_images(
        "describe",
        vec![Image::new("image/png", vec![0x89, 0x50])],
    );
    let encoded = serde_json::to_string_pretty(&conversation).unwrap();
    let decoded: Conversation = serde_json::from_str(&encoded).expect(&encoded);
    assert_eq!(decoded, conversation);
}
