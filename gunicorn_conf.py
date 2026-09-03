"""gunicorn config - start bot thread in worker process"""
import threading

def post_fork(server, worker):
    """Called after worker process has been forked. Start bot thread here."""
    from paper_bot import bot
    t = threading.Thread(target=bot.run, daemon=True)
    t.start()
    server.log.info("Bot thread started in worker process")
