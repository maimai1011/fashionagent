"""Schema definitions used by the one-time database initializer."""

USER_TABLE = """
CREATE TABLE IF NOT EXISTS user_input (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT,
                email TEXT,
                image_url TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
"""
WARDROBE_TABLE = """
   CREATE TABLE IF NOT EXISTS wardrobe_items (
                item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                raw_object_key TEXT UNIQUE NOT NULL,
                processed_object_key TEXT,
                original_filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                dominant_color TEXT,
                clothing_type TEXT,
                embedding_ref TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
"""

WARDROBE_DROP_USER_UNIQUE = """
ALTER TABLE wardrobe_items
DROP CONSTRAINT IF EXISTS wardrobe_items_user_id_key;
"""

WARDROBE_ORIGINAL_FILENAME_COLUMN = """
ALTER TABLE wardrobe_items
ADD COLUMN IF NOT EXISTS original_filename TEXT;
"""

WARDROBE_CONTENT_TYPE_COLUMN = """
ALTER TABLE wardrobe_items
ADD COLUMN IF NOT EXISTS content_type TEXT;
"""

WARDROBE_PROCESSED_OBJECT_COLUMN = """
ALTER TABLE wardrobe_items
ADD COLUMN IF NOT EXISTS processed_object_key TEXT;
"""

WARDROBE_DOMINANT_COLOR_COLUMN = """
ALTER TABLE wardrobe_items
ADD COLUMN IF NOT EXISTS dominant_color TEXT;
"""

WARDROBE_CLOTHING_TYPE_COLUMN = """
ALTER TABLE wardrobe_items
ADD COLUMN IF NOT EXISTS clothing_type TEXT;
"""

WARDROBE_EMBEDDING_REF_COLUMN = """
ALTER TABLE wardrobe_items
ADD COLUMN IF NOT EXISTS embedding_ref TEXT;
"""

WARDROBE_ERROR_COLUMN = """
ALTER TABLE wardrobe_items
ADD COLUMN IF NOT EXISTS error_message TEXT;
"""
USER_PROFILE_TABLE = """
 CREATE TABLE IF NOT EXISTS user_profiles (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                age INT NOT NULL CHECK (age BETWEEN 0 AND 130),
                gender TEXT NOT NULL CHECK (gender IN ('male', 'female')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
"""

USER_PROFILE_EMAIL_COLUMN = """
ALTER TABLE user_profiles
ADD COLUMN IF NOT EXISTS email TEXT UNIQUE;
"""

USER_PROFILE_GENDER_COLUMN = """
ALTER TABLE user_profiles
ADD COLUMN IF NOT EXISTS gender TEXT CHECK (gender IN ('male', 'female'));
"""

USER_INPUT_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS user_input_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL UNIQUE,
    session_id TEXT,
    raw_input TEXT NOT NULL,
    parsed_input JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

POSTGRES_STATEMENTS = (
    USER_TABLE,
    WARDROBE_TABLE,
    WARDROBE_DROP_USER_UNIQUE,
    WARDROBE_ORIGINAL_FILENAME_COLUMN,
    WARDROBE_CONTENT_TYPE_COLUMN,
    WARDROBE_PROCESSED_OBJECT_COLUMN,
    WARDROBE_DOMINANT_COLOR_COLUMN,
    WARDROBE_CLOTHING_TYPE_COLUMN,
    WARDROBE_EMBEDDING_REF_COLUMN,
    WARDROBE_ERROR_COLUMN,
    USER_PROFILE_TABLE,
    USER_PROFILE_EMAIL_COLUMN,
    USER_PROFILE_GENDER_COLUMN,
    USER_INPUT_LOG_TABLE,
)
