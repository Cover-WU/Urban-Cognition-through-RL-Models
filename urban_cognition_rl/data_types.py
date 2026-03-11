"""
Data types for user trajectories and visits.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd


@dataclass
class Visit:
    """
    Represents a single stay/visit at a location.

    Attributes:
    - t_start: Start timestamp of the visit
    - t_end: End timestamp of the visit
    - lon: Longitude of the location
    - lat: Latitude of the location
    - cluster_id: Cluster label (0 = missing/outside study area, -1 = HDBSCAN noise, 1+ = HDBSCAN clusters)
    - ptype: Place type (optional)
    - poi: POI information (optional)
    """
    t_start: datetime
    t_end: datetime
    lon: float
    lat: float
    cluster_id: int = 0  # 0 = missing, -1 = noise, 1+ = clusters
    ptype: Optional[int] = None
    poi: Optional[int] = None

    @property
    def duration(self) -> timedelta:
        """Calculate duration of the visit."""
        return self.t_end - self.t_start

    @property
    def date(self) -> int:
        """Extract date in YYYYMMDD format from t_start."""
        return int(self.t_start.strftime('%Y%m%d'))

    @classmethod
    def from_dict(cls, data: dict) -> 'Visit':
        """Create Visit from dictionary."""
        return cls(
            t_start=pd.to_datetime(data['t_start']),
            t_end=pd.to_datetime(data['t_end']),
            lon=float(data['lon']),
            lat=float(data['lat']),
            cluster_id=int(data.get('cluster_id', -1)),
            ptype=int(data['ptype']) if pd.notna(data.get('ptype')) else None,
            poi=int(data['poi']) if pd.notna(data.get('poi')) else None
        )

    def to_dict(self) -> dict:
        """Convert Visit to dictionary."""
        return {
            't_start': self.t_start,
            't_end': self.t_end,
            'lon': self.lon,
            'lat': self.lat,
            'cluster_id': self.cluster_id,
            'ptype': self.ptype,
            'poi': self.poi
        }


@dataclass
class Trajectory:
    """
    Represents a daily trajectory (from 3:00 AM to next day 3:00 AM).

    Attributes:
    - date: Date of the trajectory (YYYYMMDD format, representing the day starting at 3 AM)
    - visits: List of Visit objects
    """
    date: int
    visits: List[Visit] = field(default_factory=list)

    @property
    def num_visits(self) -> int:
        """Return number of visits."""
        return len(self.visits)

    @property
    def start_time(self) -> Optional[datetime]:
        """Return start time of first visit."""
        return self.visits[0].t_start if self.visits else None

    @property
    def end_time(self) -> Optional[datetime]:
        """Return end time of last visit."""
        return self.visits[-1].t_end if self.visits else None

    def add_visit(self, visit: Visit):
        """Add a visit to the trajectory."""
        self.visits.append(visit)
        self.visits.sort(key=lambda v: v.t_start)

    def to_dict(self) -> dict:
        """Convert trajectory to dictionary."""
        return {
            'date': self.date,
            'visits': [v.to_dict() for v in self.visits]
        }


class User:
    """
    Represents a user with their trajectories and memory.

    Attributes:
    - id: User identifier
    - trajectories: Dictionary mapping date to Trajectory
    - memory: User's memory (to be implemented)
    """

    DAY_THRESHOLD_HOUR = 3

    def __init__(self, id: int):
        """
        Initialize a User.

        Parameters:
        - id: User identifier
        """
        self.id = id
        self.trajectories: Dict[int, Trajectory] = {}
        self.memory: Optional[Dict] = None

    @property
    def num_trajectories(self) -> int:
        """Return number of trajectories."""
        return len(self.trajectories)

    @property
    def total_visits(self) -> int:
        """Return total number of visits across all trajectories."""
        return sum(t.num_visits for t in self.trajectories.values())

    @property
    def unique_clusters(self) -> set:
        """Return set of unique cluster IDs visited by this user.

        Note: cluster_id >= 1 are actual clusters (0 = missing, -1 = noise)
        """
        clusters = set()
        for traj in self.trajectories.values():
            for visit in traj.visits:
                if visit.cluster_id >= 1:
                    clusters.add(visit.cluster_id)
        return clusters

    def add_trajectory(self, trajectory: Trajectory):
        """Add a trajectory to the user."""
        if trajectory.date in self.trajectories:
            existing_traj = self.trajectories[trajectory.date]
            for visit in trajectory.visits:
                existing_traj.add_visit(visit)
        else:
            self.trajectories[trajectory.date] = trajectory

    def to_dataframe(self) -> pd.DataFrame:
        """Convert User object to DataFrame."""
        records = []
        for date, traj in sorted(self.trajectories.items()):
            for visit in traj.visits:
                records.append({
                    'who': self.id,
                    'date': date,
                    't_start': visit.t_start,
                    't_end': visit.t_end,
                    'lon': visit.lon,
                    'lat': visit.lat,
                    'cluster_id': visit.cluster_id,
                    'ptype': visit.ptype,
                    'poi': visit.poi
                })
        return pd.DataFrame(records)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Dict[int, 'User']:
        """Create User objects from DataFrame."""
        df = df.sort_values(['who', 't_start']).reset_index(drop=True)
        users = {}

        for who, group in df.groupby('who'):
            user = cls(int(who))
            current_date = None
            current_traj = None

            for _, row in group.iterrows():
                t_start = pd.to_datetime(row['t_start'])
                t_end = pd.to_datetime(row['t_end'])

                date = int(row['date'])

                if date != current_date:
                    if current_traj is not None:
                        user.add_trajectory(current_traj)
                    current_traj = Trajectory(date=date)
                    current_date = date

                visit = Visit(
                    t_start=t_start,
                    t_end=t_end,
                    lon=float(row['lon']),
                    lat=float(row['lat']),
                    cluster_id=int(row.get('cluster_id', 0)) if pd.notna(row.get('cluster_id')) else 0,
                    ptype=int(row['ptype']) if pd.notna(row.get('ptype')) else None,
                    poi=int(row['poi']) if pd.notna(row.get('poi')) else None
                )
                current_traj.add_visit(visit)

            if current_traj is not None:
                user.add_trajectory(current_traj)

            users[int(who)] = user

        return users

    def get_trajectory_split_by_day(self, gap_minutes: int = 180) -> List:
        """Split trajectory by day with gap threshold."""
        result = []
        last_visit = None
        last_threshold_date = None

        for date, traj in sorted(self.trajectories.items()):
            for visit in traj.visits:
                if last_visit is not None:
                    time_diff = (visit.t_start - last_visit.t_end).total_seconds() / 60

                    if time_diff > gap_minutes or date != last_threshold_date:
                        if last_visit is not None:
                            result.append((last_visit, last_threshold_date))
                        last_visit = visit
                        last_threshold_date = date
                    else:
                        last_visit = visit
                        last_threshold_date = date
                else:
                    last_visit = visit
                    last_threshold_date = date

        if last_visit is not None:
            result.append((last_visit, last_threshold_date))

        return result

    def __repr__(self) -> str:
        return f"User(id={self.id}, trajectories={self.num_trajectories}, total_visits={self.total_visits})"
