from apscheduler.schedulers.background import BackgroundScheduler
from app.core.config import get_settings
from app.core.evaluation import run_evaluation

settings = get_settings()


def start_scheduler():
    sched = BackgroundScheduler(timezone="Asia/Kolkata")
    sched.add_job(
        run_evaluation, "cron",
        hour=settings.eval_cron_hour, minute=settings.eval_cron_minute,
        id="daily_evaluation", replace_existing=True,
    )
    sched.start()
    return sched
