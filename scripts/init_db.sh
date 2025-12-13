#!/bin/bash
# Database initialization script for Memlayer
# This script runs database migrations when the container starts

set -e

echo "🔧 Initializing Memlayer database..."

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
until PGPASSWORD="${POSTGRES_PASSWORD}" psql -h postgres -U memlayer -d memlayer -c '\q' 2>/dev/null; do
  echo "   PostgreSQL is unavailable - sleeping"
  sleep 1
done

echo "✅ PostgreSQL is ready!"

# Run migrations
echo "📦 Running database migrations..."
PGPASSWORD="${POSTGRES_PASSWORD}" psql -h postgres -U memlayer -d memlayer -f /app/migrations/001_create_salience_tables.sql || {
  echo "⚠️  Migration already applied or failed - continuing..."
}

echo "✅ Database initialization complete!"
