HOST=${HOST:-0.0.0.0}
PORT=${PORT:-5000}
WORKERS=${WORKERS:-2}
TIMEOUT=${TIMEOUT:-120}

echo "Starting Gunicorn..."
exec gunicorn -w $WORKERS -b $HOST:$PORT --timeout $TIMEOUT api.app:app
