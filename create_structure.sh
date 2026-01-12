#!/usr/bin/env bash
set -e

# Root-level files (touch creates empty placeholders)
touch README.md docker-compose.yml requirements.txt .env.example .gitignore \
      Dockerfile pyproject.toml .pre-commit-config.yaml

# app/
mkdir -p app/{api/{middleware,routers/v1},models,services,agents,mcp,database/{migrations/versions,chroma,repositories},integrations/auth,utils,monitoring}
touch app/main.py app/config.py
touch app/api/dependencies.py
touch app/api/middleware/{auth.py,logging.py,rate_limiter.py,error_handler.py}
touch app/api/routers/v1/{memory.py,agents.py,mcp.py,websockets.py}
touch app/api/routers/health.py
touch app/models/{database.py,schemas.py,enums.py,types.py}
touch app/services/{memory_service.py,agent_service.py,orchestrator.py,llm_service.py,embedding_service.py,context_manager.py,message_queue.py,cache_service.py,auth_service.py}
touch app/agents/{base_agent.py,email_agent.py,calendar_agent.py,slack_agent.py,registry.py}
touch app/mcp/{server.py,client.py,tools.py,registry.py,validators.py}
touch app/database/connection.py app/database/models.py app/database/chroma_handler.py
touch app/database/migrations/{alembic.ini,env.py}
touch app/database/repositories/{agent_repository.py,memory_repository.py,context_repository.py}
touch app/integrations/{client.py,slack_connector.py,twilio_connector.py,zendesk_connector.py}
touch app/integrations/auth/{oauth2.py,api_key.py}
touch app/utils/{logging.py,security.py,helpers.py,exceptions.py,validators.py,circuit_breaker.py}
touch app/monitoring/{metrics.py,health_check.py,tracing.py,alerts.py}

# ui/streamlit_app/
mkdir -p ui/streamlit_app/{pages,assets}
touch ui/streamlit_app/{app.py,requirements.txt}
touch ui/streamlit_app/pages/{Dashboard.py,Settings.py}

# tests/
mkdir -p tests/{fixtures,test_agents,test_services,test_api/{v1},test_integration,test_contracts,test_performance,test_security}
touch tests/conftest.py

# pipedream/
mkdir -p pipedream/utils
touch pipedream/{email-connector.js,calendar-connector.js,database-connector.js,slack-connector.js}
touch pipedream/utils/{auth.js,validation.js,helpers.js}

# infrastructure/
mkdir -p infrastructure/{terraform,kubernetes}

# scripts/
mkdir -p scripts
touch scripts/{setup.py,deploy.py,seed_data.py,backup.py,migrate.py,performance_test.py}

# docs/
mkdir -p docs/{diagrams,monitoring}
touch docs/{README.md,architecture.md,api.md,deployment.md,development.md,user_guide.md,security.md,troubleshooting.md,contributing.md}

# data/
mkdir -p data/{sqlite,logs,exports,backups,cache}

# configs/
mkdir -p configs
touch configs/{development.env,staging.env,production.env,logging.yaml}

# .github/
mkdir -p .github/{workflows,ISSUE_TEMPLATE}
touch .github/workflows/{ci.yml,deploy.yml,tests.yml,security.yml,performance.yml}
touch .github/ISSUE_TEMPLATE/{bug_report.md,feature_request.md}
touch .github/pull_request_template.md

echo "Folder structure created successfully."
