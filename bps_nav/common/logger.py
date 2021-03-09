import logging

class BPSNavLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().addHandler(logging.StreamHandler())

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        self.addHandler(filehandler)

logger = BPSNavLogger(name="bps-nav")
