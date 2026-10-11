"""Schema definitions used by the one-time database initializer."""

USER_PROFILE_TABLE = """
 CREATE TABLE IF NOT EXISTS user_profiles (
                id SERIAL PRIMARY KEY,
                user_id TEXT UNIQUE NOT NULL,
                name TEXT,
                email TEXT UNIQUE,
                age INT,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
"""

USER_INPUT_TABLE = """
CREATE TABLE IF NOT EXISTS user_input (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    raw_object_key TEXT UNIQUE NOT NULL,
    original_filename TEXT NOT NULL,
    content_type TEXT,
    status TEXT NOT NULL DEFAULT 'uploaded',
    clothing_type TEXT,
    classification_confidence DOUBLE PRECISION,
    dominant_color TEXT,
    embedding_ref TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT user_input_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);
"""

POSTGRES_STATEMENTS = (
    USER_PROFILE_TABLE,
    USER_INPUT_TABLE,
)
