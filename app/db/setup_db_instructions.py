"""
Supabase DB Setup Instructions

Run the following SQL script manually in the Supabase SQL Editor
to create the required `tasks` table and enable policies + realtime.
"""

from app.core.logging import logger

create_tasks_table_sql = """
-- Enable gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    result TEXT,
    agent_type TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for faster lookup by user_id
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);

-- Enable Row-Level Security (RLS)
ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;

-- Policy: allow users to read only their tasks
CREATE POLICY tasks_policy_select
  ON tasks FOR SELECT
  USING ((select auth.uid()) = user_id);

-- Policy: allow users to insert their own tasks
CREATE POLICY tasks_policy_insert
  ON tasks FOR INSERT
  WITH CHECK ((select auth.uid()) = user_id);

-- Policy: allow users to update their own tasks
CREATE POLICY tasks_policy_update
  ON tasks FOR UPDATE
  USING ((select auth.uid()) = user_id);

-- Enable realtime for this table
ALTER PUBLICATION supabase_realtime ADD TABLE tasks;
"""

def print_setup_instructions():
    logger.info("Run the following SQL manually in Supabase SQL Editor:\n\n%s", create_tasks_table_sql)

if __name__ == "__main__":
    print_setup_instructions()
