"""Worker process for processing pipeline jobs from Redis."""
import time
import logging
import signal

logger = logging.getLogger(__name__)

class Worker:
    def __init__(self):
        self.running = True
        # TODO: Initialize Redis connection
        # self.redis = Redis(...)

    def _handle_shutdown(self, sig, frame):
        logger.info("Graceful shutdown initiated...")
        self.running = False

    def run(self):
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        logger.info("Worker started. Polling for jobs...")
        
        while self.running:
            # TODO: Poll Redis for jobs (e.g. blpop)
            # job = self.redis.blpop("m2a_queue", timeout=1)
            job = None
            
            if job:
                logger.info(f"Picked up job: {job}")
                # 1. Update job status to 'running'
                # 2. Run PipelineGraph
                # 3. Update job status to 'completed' or 'failed'
            
            time.sleep(1) # Stub for polling interval
            
        logger.info("Worker stopped.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = Worker()
    worker.run()
