"""SQL statements for raw user data storage."""

INSERT_USER_PROFILE = """
INSERT INTO user_profiles (
    user_id,
    email,
    age,
    gender
)
VALUES (%s, %s, %s, %s)
ON CONFLICT (user_id) DO NOTHING
RETURNING id, created_at
"""

INSERT_USER_IMAGE = """
INSERT INTO wardrobe_items (
    user_id,
    raw_object_key,
    original_filename,
    content_type,
    status
)
VALUES (%s, %s, %s, %s, 'uploaded')
RETURNING item_id
"""

SELECT_WARDROBE_IMAGE = """
SELECT item_id, user_id, raw_object_key, content_type
FROM wardrobe_items
WHERE item_id = %s
"""

SELECT_WARDROBE_RESULT = """
SELECT item_id, user_id, status, original_filename, processed_object_key,
       dominant_color, clothing_type, error_message, raw_object_key, content_type
FROM wardrobe_items
WHERE item_id = %s AND user_id = %s
"""

SELECT_WARDROBE_RESULTS = """
SELECT item_id, user_id, status, original_filename, processed_object_key,
       dominant_color, clothing_type, error_message, raw_object_key, content_type
FROM wardrobe_items
WHERE user_id = %s
ORDER BY created_at DESC NULLS LAST, item_id DESC
"""

UPDATE_WARDROBE_STATUS = """
UPDATE wardrobe_items
SET status = %s,
    error_message = %s,
    updated_at = CURRENT_TIMESTAMP
WHERE item_id = %s
"""

COMPLETE_WARDROBE_PROCESSING = """
UPDATE wardrobe_items
SET status = 'completed',
    processed_object_key = %s,
    dominant_color = %s,
    clothing_type = %s,
    embedding_ref = %s,
    error_message = NULL,
    updated_at = CURRENT_TIMESTAMP
WHERE item_id = %s
"""
