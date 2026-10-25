# Emotion-Based Telugu Recommendation

## Run
    pip install -r requirements.txt
    python app.py          # open http://localhost:5000

## Live recommendations (optional)
Without a key the app uses the built-in lists in app.py.
To fetch fresh videos per emotion from YouTube:

1. Google Cloud Console -> create a project -> enable "YouTube Data API v3"
   -> Credentials -> Create API key.
2. Set it as an environment variable (never paste it into the code):
       Windows (cmd):   set YOUTUBE_API_KEY=your_key
       PowerShell:      $env:YOUTUBE_API_KEY="your_key"
       Mac / Linux:     export YOUTUBE_API_KEY=your_key
3. Run `python app.py` in the same terminal.

Results are cached for 1 hour per emotion (a search costs 100 of the 10,000
free daily quota units). If the key is missing or YouTube fails, the app
falls back to the built-in lists automatically.
Edit SEARCH_QUERIES in app.py to change what is searched for each emotion.
