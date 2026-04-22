# Deployment

## Backend: Cloud Run

1. Build and deploy the FastAPI backend to Cloud Run.
2. Set `NOVA_RL_CORS_ORIGINS` to your Firebase Hosting origin, for example:
   - `https://nova-rl.web.app,https://nova-rl.firebaseapp.com`
3. After deploy, copy the Cloud Run service URL:
   - `https://nova-rl-backend-460592928457.asia-south1.run.app`

Example:

```powershell
gcloud run deploy nova-rl-backend `
  --source . `
  --region asia-south1 `
  --allow-unauthenticated `
  --timeout 300 `
  --set-env-vars "NOVA_RL_CORS_ORIGINS=https://nova-rl.web.app,https://nova-rl.firebaseapp.com"
```

## Frontend: Firebase Hosting

1. Set the backend URL in `ui/public/index.html` using the `nova-api-base` meta tag, or set `window.localStorage.NOVA_API_BASE`.
2. Initialize Firebase Hosting:

```powershell
firebase login
firebase init hosting
```

3. Deploy the static frontend:

```powershell
firebase deploy --only hosting:app
```

## Optional Rewrite

If you want `/api/**` on the Firebase domain to proxy to Cloud Run, add a Hosting rewrite using your Cloud Run `serviceId` and `region`.

## Verification

1. Open the Firebase Hosting URL: `https://nova-rl.web.app`
2. Confirm the provider list loads.
3. Complete the workflow:
   - Step 0 configuration
   - Step 1 initialize session
   - Step 2 select task
   - Step 3 fetch state
   - Step 4 execute action
   - Step 5 inspect results
