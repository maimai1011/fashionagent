"""SQL statements for raw user data storage."""

INSERT_USER_PROFILE = """
INSERT INTO user_profiles (
    user_id,
    name,
    email,
    age,
    preferences
)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (user_id) DO NOTHING
RETURNING id, created_at
"""

INSERT_USER_IMAGE = """
INSERT INTO user_input (
    user_id,
    raw_object_key,
    original_filename,
    content_type
)
VALUES (%s, %s, %s, %s)
RETURNING id
"""
