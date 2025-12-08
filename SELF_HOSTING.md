# Memlayer - Self-Hosted Setup 🚀

Host Memlayer for yourself and your friends with complete control!

## Quick Start (5 minutes)

### Prerequisites
- Docker & Docker Compose
- OpenAI API key (or use local models)
- 2GB RAM minimum

### 1. Clone & Configure

```bash
git clone https://github.com/thebnbrkr/memlayer.git
cd memlayer

# Create environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Start Everything

```bash
# Start all services (PostgreSQL, Memlayer API, optional Memgraph)
docker-compose up -d

# Check logs
docker-compose logs -f memlayer
```

### 3. Test It Works

```bash
# Test the API
curl http://localhost:8000/health

# Or visit in browser
open http://localhost:8000/docs
```

Done! Memlayer is running at `http://localhost:8000`

---

## Architecture

```
┌─────────────────┐
│   Your Friends  │
│   (API clients) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Memlayer API   │ ← http://localhost:8000
│  (FastAPI)      │
└────────┬────────┘
         │
    ┌────┴────┐
    v         v
┌─────────┐ ┌──────────┐
│ Postgres│ │ Memgraph │
│ (vector)│ │ (graph)  │
└─────────┘ └──────────┘
```

---

## Configuration

### Multi-User Setup

Each friend gets their own tenant_id:

```python
# Friend 1: Alice
alice_client = OpenAI(
    api_key="your-openai-key",
    user_id="alice",
    tenant_id="alice",
    salience_config=alice_config
)

# Friend 2: Bob
bob_client = OpenAI(
    api_key="your-openai-key",
    user_id="bob",
    tenant_id="bob",
    salience_config=bob_config
)
```

**Important:** For now, their memories are isolated in **vector DB only**. Graph entities are shared (see Limitations below).

### Custom Salience Configs

Each friend can have their own salience config:

```python
from memlayer import TenantSalienceConfig
from memlayer.config.salience import SalienceComponent, ScoringFunctionType

# Alice wants strict filtering
alice_config = TenantSalienceConfig(
    tenant_id="alice",
    config_name="strict",
    components=[...],
    threshold_config={"strategy": "absolute", "threshold": 0.7}
)

# Bob wants permissive filtering
bob_config = TenantSalienceConfig(
    tenant_id="bob",
    config_name="permissive",
    components=[...],
    threshold_config={"strategy": "absolute", "threshold": 0.2}
)
```

---

## API Usage

### FastAPI Endpoints

Access the API docs at: `http://localhost:8000/docs`

**Key endpoints:**
- `POST /api/config/salience` - Create salience config
- `GET /api/config/salience` - List configs
- `POST /api/config/salience/{name}/test` - Test a config

### Python Client

```python
from memlayer import OpenAI, TenantSalienceConfig

# Create config via API
import requests
response = requests.post(
    "http://localhost:8000/api/config/salience",
    json={...config...}
)

# Or use client directly
client = OpenAI(
    model="gpt-4o-mini",
    user_id="alice",
    salience_config=my_config,
    storage_path="./data/alice"
)

# Chat with memory
response = client.chat([
    {"role": "user", "content": "Remember that I love Python"}
])

# Get salience logs
logs = client.get_salience_logs()
for log in logs:
    print(f"{log['decision']}: {log['fact']}")
```

---

## Security

### For Friends (Low Security)

If you're just hosting for friends on your local network:

```yaml
# docker-compose.yml
services:
  memlayer:
    ports:
      - "8000:8000"  # Only accessible on your network
    environment:
      - SIMPLE_API_KEYS=alice:key1,bob:key2  # Basic auth
```

### For Production (High Security)

If you want to host publicly:

1. **Add Authentication**
   ```bash
   # Generate API keys
   python scripts/generate_api_keys.py
   ```

2. **Enable HTTPS**
   ```yaml
   # Use Caddy or Nginx
   services:
     caddy:
       image: caddy:2
       ports:
         - "443:443"
       volumes:
         - ./Caddyfile:/etc/caddy/Caddyfile
   ```

3. **Add Rate Limiting**
   ```python
   # Already built-in, just configure:
   RATE_LIMIT=100/hour
   ```

---

## Storage

### Data Persistence

All data is stored in Docker volumes:

```bash
# Backup data
docker-compose exec postgres pg_dump memlayer > backup.sql

# Restore data
docker-compose exec -T postgres psql memlayer < backup.sql
```

### Storage Locations

- **Vector DB (ChromaDB):** `./data/chroma/`
- **Graph DB (NetworkX):** `./data/graphs/`
- **PostgreSQL:** Docker volume `memlayer_postgres`

---

## Scaling

### For 1-10 Friends

Current setup is fine:
- Single server
- ChromaDB for vectors
- NetworkX for graphs

### For 10+ Users

Upgrade to:
- **Vector:** Switch to Qdrant or Zilliz (cloud or self-hosted)
- **Graph:** Switch to Memgraph (included in docker-compose)
- **Web server:** Use Gunicorn with multiple workers

```bash
# Start with Memgraph
docker-compose --profile production up -d
```

---

## Monitoring

### Basic Health Check

```bash
# Check if everything is running
curl http://localhost:8000/health

# View logs
docker-compose logs -f memlayer
```

### Advanced Monitoring

Enable Prometheus metrics:

```yaml
# docker-compose.yml
services:
  memlayer:
    environment:
      - ENABLE_METRICS=true

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

---

## Limitations (Current)

### ⚠️ Graph Data Not Fully Isolated

**Problem:** Alice and Bob share the same knowledge graph.
- If Alice adds entity "TechCorp", Bob can see it too
- Vector memories ARE isolated (Alice can't see Bob's facts)

**Workaround:** Use separate storage paths for each user:
```python
alice_client = OpenAI(storage_path="./data/alice")
bob_client = OpenAI(storage_path="./data/bob")
```

**Proper Fix:** Add tenant_id to graph operations (coming soon)

### Limited to Single Server

Current setup runs on one machine. For high availability, you'd need:
- Load balancer
- Multiple Memlayer instances
- Shared PostgreSQL/Memgraph

---

## Troubleshooting

### "Connection refused" error

```bash
# Check if services are running
docker-compose ps

# Restart everything
docker-compose restart
```

### "Out of memory" error

```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 4GB
```

### Slow responses

```bash
# Check OpenAI API usage
# You might be hitting rate limits

# Use local embeddings instead
OPERATION_MODE=local
```

---

## Updating

```bash
# Pull latest code
git pull origin main

# Rebuild containers
docker-compose build --no-cache

# Restart
docker-compose up -d
```

---

## Cost Estimation

### Self-Hosting Costs (Monthly)

**For 5 friends, light usage:**
- Server: $5-10/month (DigitalOcean, Hetzner)
- OpenAI API: $5-20/month (depends on usage)
- **Total: ~$15-30/month**

**For 20 users, moderate usage:**
- Server: $20-40/month (2 vCPU, 4GB RAM)
- OpenAI API: $50-100/month
- **Total: ~$70-140/month**

### Compare to Hosted SaaS

If you built a hosted SaaS:
- Server costs: $50-200/month
- Database: $25-100/month
- Monitoring: $20-50/month
- **Total: ~$95-350/month (before users even pay you!)**

**Self-hosting is much cheaper for friends!**

---

## Support

- **Issues:** https://github.com/thebnbrkr/memlayer/issues
- **Docs:** https://github.com/thebnbrkr/memlayer/tree/main/docs
- **Examples:** https://github.com/thebnbrkr/memlayer/tree/main/examples

---

## License

MIT - You can use it however you want! Commercial, personal, friends, everyone.

---

**Happy self-hosting! 🚀**
