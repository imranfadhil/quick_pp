#!/bin/sh
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h postgres -p 5432 -U ${POSTGRES_USER}; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 2
done
echo "PostgreSQL is up - executing database initialization"
python /code/quick_pp/app/docker/init_db.py

echo "Starting application"
exec python main.py app
