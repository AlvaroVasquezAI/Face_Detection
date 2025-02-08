# src/feedback/collector.py
import os
import json
from .metrics import calculate_feedback_metrics

class FeedbackCollector:
    def __init__(self, feedback_dir="feedback"):
        self.feedback_dir = feedback_dir
        os.makedirs(feedback_dir, exist_ok=True)
        self.feedback_file = os.path.join(feedback_dir, "feedback_data.json")
        self.load_feedback()
    
    def load_feedback(self):
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                self.feedback_data = json.load(f)
        else:
            self.feedback_data = []
    
    def add_feedback(self, feedback):
        self.feedback_data.append(feedback)
        self.save_feedback()
        self.update_metrics()
    
    def save_feedback(self):
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=4)
    
    def update_metrics(self):
        metrics = calculate_feedback_metrics(self.feedback_data)
        with open(os.path.join(self.feedback_dir, "feedback_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)