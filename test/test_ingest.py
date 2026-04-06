from ingestion.pipeline import EduQueryIngester
ingester = EduQueryIngester()
segs = ingester.ingest_video(r"dataset\#3 Machine Learning Specialization [Course 1, Week 1, Lesson 2] [XtlwSmJfUs4].mp4","test","vid01","processed_videos")
assert len(segs) > 0
print("OK", len(segs))