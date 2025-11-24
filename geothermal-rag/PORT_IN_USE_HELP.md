# Port Already in Use - Solutions

## Problem

```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 7860)
OSError: Cannot find empty port in range: 7860-7860
```

This means another instance of the application is already running on port 7860.

---

## Quick Solutions

### Option 1: Use the Stop Script (Easiest)

**Windows:**
```batch
stop_app.bat
```

**Linux/Mac:**
```bash
chmod +x stop_app.sh
./stop_app.sh
```

Then restart the app:
```bash
python app.py
```

---

### Option 2: Automatic Port Finding (New Feature!)

The app now automatically tries ports 7860-7869. Just run it again:

```bash
python app.py
```

It will find an available port and tell you which one it's using.

---

### Option 3: Manual Port Check and Kill

**Windows:**

1. Find the process:
   ```batch
   netstat -ano | findstr :7860
   ```
   
2. Note the PID (last column)

3. Kill the process:
   ```batch
   taskkill /PID <PID> /F
   ```
   Replace `<PID>` with the actual number

**Linux/Mac:**

1. Find the process:
   ```bash
   lsof -ti:7860
   ```

2. Kill the process:
   ```bash
   kill -9 <PID>
   ```
   Or simply:
   ```bash
   lsof -ti:7860 | xargs kill -9
   ```

---

### Option 4: Use a Different Port

**Method A: Environment Variable**
```batch
# Windows
set GRADIO_SERVER_PORT=7861
python app.py

# Linux/Mac
export GRADIO_SERVER_PORT=7861
python app.py
```

**Method B: Edit Config File**

Edit `config/config.yaml`:
```yaml
ui:
  port: 7861  # Change from 7860
  share: false
  server_name: "0.0.0.0"
```

---

## Prevention

### Close Previous Instances Before Starting New Ones

**Windows:**
- Press `Ctrl+C` in the terminal running the app
- Or close the terminal window

**Linux/Mac:**
- Press `Ctrl+C` in the terminal running the app

### Check Running Instances

**Windows:**
```batch
netstat -ano | findstr :7860
```

**Linux/Mac:**
```bash
lsof -i:7860
```

---

## Common Scenarios

### Scenario 1: "I closed the terminal but port is still busy"

The process might still be running in the background. Use Option 1 or 3 above.

### Scenario 2: "I want to run multiple instances"

Use Option 4 to assign different ports:
- Instance 1: Port 7860
- Instance 2: Port 7861
- Instance 3: Port 7862

### Scenario 3: "Another application is using port 7860"

Either:
- Stop that application
- Or use Option 4 to change the port for this app

---

## Testing

After fixing the issue, verify the app starts:

```bash
python app.py
```

Expected output:
```
INFO:__main__:✓ System initialized successfully
INFO:__main__:Attempting to start server on port 7860...
Running on local URL:  http://0.0.0.0:7860
```

Or if port 7860 was busy:
```
INFO:__main__:Port 7860 is already in use.
INFO:__main__:Attempting to find an available port automatically...
INFO:__main__:Trying port 7861...
INFO:__main__:✓ Successfully started on port 7861
Running on local URL:  http://0.0.0.0:7861
```

---

## Still Having Issues?

1. **Restart your computer** - This will close all background processes
2. **Check firewall settings** - Ensure ports 7860-7869 are not blocked
3. **Use localhost** - Access via `http://localhost:7860` instead of `0.0.0.0:7860`
4. **Check antivirus** - Some antivirus software blocks port binding

---

## Need Help?

If none of these solutions work:

1. Check the full error message in the terminal
2. Look for any Python traceback details
3. Verify all dependencies are installed: `pip install -r requirements.txt`
4. Check Ollama is running: `ollama serve`

---

**The app is now more resilient and will automatically find available ports!** 🚀
