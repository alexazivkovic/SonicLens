#!/bin/bash

npx concurrently \
  -n backend,frontend \
  -c blue,green \
  --kill-others \
  "cd backend && .venv/bin/uvicorn app.main:app --port 8000" \
  "cd client && npm run dev"