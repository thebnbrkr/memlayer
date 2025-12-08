# Memlayer Docker Quick Start

Get Memlayer running in 3 minutes with Docker Compose.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Quick Start

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/thebnbrkr/memlayer.git
cd memlayer

# Create .env file from example
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

**Required:** Set your OpenAI API key in `.env`:
```bash
OPENAI_API_KEY=sk-...your-key-here...
```

### 2. Start Services

```bash
# Start PostgreSQL + Memlayer API
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 3. Verify It's Running

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "service": "memlayer-api",
#   "version": "1.0.0",
#   ...
# }
```

### 4. View API Documentation

Open in your browser:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## What's Running?

After `docker-compose up -d`, you have:

1. **PostgreSQL** (port 5432)
   - Stores salience configurations and audit logs
   - Credentials: `memlayer` / `changeMe123!` (change in production!)

2. **Memlayer API** (port 8000)
   - FastAPI server with salience configuration endpoints
   - Interactive docs at `/docs`

3. **Persistent Volumes**
   - `postgres_data` - Database storage
   - `./data` - Vector embeddings and graph data

## Next Steps

### Create Your First Salience Config

```bash
curl -X POST "http://localhost:8000/api/config/salience/" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "my_org",
    "config_name": "tech_focused",
    "components": [
      {
        "name": "technical_depth",
        "weight": 1.0,
        "scoring_function": "keyword_match",
        "scoring_config": {
          "keywords": ["python", "code", "algorithm", "AI"]
        }
      }
    ],
    "threshold_config": {
      "strategy": "absolute",
      "absolute_threshold": 0.5
    },
    "decision_rules": []
  }'
```

### Test Your Config

```bash
curl -X POST "http://localhost:8000/api/config/salience/tech_focused/test?tenant_id=my_org" \
  -H "Content-Type: application/json" \
  -d '{
    "fact": "I am learning Python to build AI applications"
  }'

# Returns salience score, decision (STORE/SKIP), and component scores
```

### Use in Python Code

```python
from memlayer import OpenAI, TenantSalienceConfig
from memlayer.config.salience import SalienceComponent, ScoringFunctionType

# Create custom config
my_config = TenantSalienceConfig(
    tenant_id="alice",
    config_name="my_config",
    components=[
        SalienceComponent(
            name="technical",
            weight=1.0,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["python", "code"]}
        )
    ],
    threshold_config={"strategy": "absolute", "absolute_threshold": 0.5}
)

# Use with OpenAI client
client = OpenAI(
    user_id="alice",
    tenant_id="alice",
    storage_path="./data",
    salience_config=my_config
)

response = client.chat([
    {"role": "user", "content": "I love coding in Python!"}
])

# Check what was stored
print(client.get_salience_logs())
```

## Configuration

Edit `.env` to customize:

- **OPERATION_MODE**: `online` (OpenAI embeddings), `local` (sentence-transformers), `lightweight` (keywords only)
- **RATE_LIMIT**: API rate limiting (e.g., `100/hour`)
- **DEBUG_MODE**: Set to `true` for detailed logs
- **STORAGE_PATH**: Where to store vector/graph data

See `.env.example` for all options.

## Multi-User Setup

To host for multiple users:

1. **Use different `tenant_id` and `user_id` for each user:**
   ```python
   alice = OpenAI(user_id="alice", tenant_id="my_org", ...)
   bob = OpenAI(user_id="bob", tenant_id="my_org", ...)
   ```

2. **Data is automatically isolated:**
   - Vector memories are filtered by `user_id`
   - Each user has their own memory store

3. **Current limitation:** Graph entities are global (not isolated by tenant)
   - See `SELF_HOSTING.md` for details and workarounds

## Production Deployment

For production use:

1. **Change database password** in `.env`:
   ```bash
   POSTGRES_PASSWORD=your-strong-password-here
   ```

2. **Set CORS origins** to your domain:
   ```bash
   CORS_ORIGINS=https://yourapp.com
   ```

3. **Use HTTPS:** Set up reverse proxy (nginx/Caddy) with SSL certificates

4. **Enable production backends** (optional):
   - Uncomment Memgraph service in `docker-compose.yml` for production graph storage
   - Uncomment Qdrant service for production vector storage
   - Or configure external services (Pinecone, Zilliz, etc.) in `.env`

See `SELF_HOSTING.md` for complete production guide.

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs memlayer

# Common issue: missing OPENAI_API_KEY
# Fix: Add your key to .env file
```

### Database connection error
```bash
# Restart services
docker-compose restart

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Port already in use
```bash
# Change ports in docker-compose.yml:
ports:
  - "8001:8000"  # Change 8000 to 8001 or any free port
```

## Stopping Services

```bash
# Stop containers (keeps data)
docker-compose stop

# Stop and remove containers (keeps data volumes)
docker-compose down

# Remove everything including data (⚠️ DELETES ALL DATA)
docker-compose down -v
```

## Useful Commands

```bash
# View logs
docker-compose logs -f memlayer
docker-compose logs -f postgres

# Restart a service
docker-compose restart memlayer

# Access PostgreSQL shell
docker-compose exec postgres psql -U memlayer -d memlayer

# Access container shell
docker-compose exec memlayer bash

# Check running containers
docker-compose ps

# Update to latest code
git pull
docker-compose build
docker-compose up -d
```

## Support

- **Documentation:** See `SELF_HOSTING.md` for complete guide
- **Issues:** https://github.com/thebnbrkr/memlayer/issues
- **Examples:** Check the `examples/` directory

## Cost Estimate

Running for yourself + 5 friends:

- **DigitalOcean Droplet (2GB RAM):** $12/month
- **PostgreSQL storage:** ~$3/month (20GB)
- **Total:** ~$15/month

See `SELF_HOSTING.md` for hosting platform comparisons.
