"""Bounded paid-request concurrency with provider-429 feedback."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading,time
from core import save


class TTSLimit:
    def __init__(self,target=15):
        self.target=target;self.limit=target;self.active=0;self.peak=0;self.rejections=0
        self.cv=threading.Condition();self.last_save=0

    def report(self,force=False):
        if force or time.time()-self.last_save>5:
            self.last_save=time.time()
            save('concurrency.json',{'configured_tts_workers':self.target,'effective_limit':self.limit,
                'active_requests':self.active,'peak_active_requests':self.peak,'rate_limit_rejections':self.rejections,
                'max_queued_and_active_tts_tasks':90,'updated_at':time.time()})

    @contextmanager
    def slot(self,check):
        with self.cv:
            while self.active>=self.limit:self.cv.wait(.5);check()
            check();self.active+=1;self.peak=max(self.peak,self.active);self.report()
        try:yield
        finally:
            with self.cv:self.active-=1;self.cv.notify_all();self.report()

    def rejected(self):
        with self.cv:
            self.rejections+=1;self.limit=max(1,self.limit-1);self.report(True);self.cv.notify_all()


class BoundedPool:
    def __init__(self,workers,check,max_pending=90):
        self.pool=ThreadPoolExecutor(max_workers=workers);self.slots=threading.Semaphore(max_pending);self.check=check
    def __enter__(self):return self
    def __exit__(self,*args):return self.pool.__exit__(*args)
    def submit(self,fn,*args,**kwargs):
        while not self.slots.acquire(timeout=.5):self.check()
        try:
            self.check();f=self.pool.submit(fn,*args,**kwargs)
        except BaseException:self.slots.release();raise
        f.add_done_callback(lambda _:self.slots.release())
        return f
