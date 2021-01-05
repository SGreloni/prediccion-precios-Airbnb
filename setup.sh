mkdir -p ~/.streamlit/
echo "[general]
email = \"santiagogreloni@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml