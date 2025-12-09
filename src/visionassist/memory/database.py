import json
from sqlalchemy import create_engine, select, delete, func
from sqlalchemy.orm import sessionmaker
from visionassist.config import MAX_OBJECT_ENTRIES
from .models import Base, Detection, DetectedObject


class Database:
    def __init__(self, db_path="objects.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def insert_detection(self, image_path, objects):
        with self.SessionLocal() as db:
            detection = Detection(image_path=image_path)
            db.add(detection)
            db.flush()

            for obj in objects:
                db.add(
                    DetectedObject(
                        detection_id=detection.id,
                        object_name=obj.get("label"),
                        confidence=obj.get("confidence"),
                        bbox=json.dumps(obj.get("bbox", {}))
                    )
                )
                # enforce at most MAX_OBJECT_ENTRIES entries per object
                self._prune_objects(db, obj["label"], max_entries=MAX_OBJECT_ENTRIES)

            db.commit()

    def _prune_objects(self, db, object_name, max_entries=50):
        """
        Deletes older entries keeping only last 'max_entries' rows per object.
        """
        # get IDs of latest allowed items
        stmt = (
            select(DetectedObject.id)
            .where(DetectedObject.object_name == object_name)
            .order_by(DetectedObject.timestamp.desc())
            .limit(max_entries)
        )
        keep_ids = [row[0] for row in db.execute(stmt).all()]

        # delete older ones
        del_stmt = (
            delete(DetectedObject)
            .where(DetectedObject.object_name == object_name)
            .where(DetectedObject.id.not_in(keep_ids))
        )
        db.execute(del_stmt)

    def get_latest_objects(self, object_name, limit=10):
        with self.SessionLocal() as db:
            stmt = (
                select(DetectedObject)
                .where(DetectedObject.object_name == object_name)
                .order_by(DetectedObject.timestamp.desc())
                .limit(limit)
            )
            return db.execute(stmt).scalars().all()

    def get_all_detections(self):
        with self.SessionLocal() as db:
            stmt = select(Detection).order_by(Detection.timestamp.desc())
            return db.execute(stmt).scalars().all()

    def get_detected_object_count(self):
        with self.SessionLocal() as db:
            stmt = func.count(DetectedObject.id)
            return db.execute(stmt).scalar()
