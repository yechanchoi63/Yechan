from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import simpy

# Reuse the existing CSV schema and loaders.
from .sim import (
    Step,
    can_assemble,
    consume_children,
    due_by_month,
    expand_required_leaves,
    load_bom,
    load_molds,
    load_routing,
    read_csv,
    working_minutes,
)


@dataclass(eq=False)
class SimJob:
    """One physical unit progressing through its routing."""

    job_id: int
    item: str
    step_index: int = 0
    ready_at: float = 0.0
    held_molds: set[str] = field(default_factory=set)


def load_resource_counts() -> Counter:
    """Read actual resource counts; do not override oven capacity."""

    counts = Counter()

    for row in read_csv("resources.csv"):
        name = row["resource"]
        count = int(row["count"])

        if name in counts:
            raise ValueError(f"Duplicate resource: {name}")

        if count < 0:
            raise ValueError(f"Negative resource count: {name}")

        counts[name] = count

    return counts


class Factory:
    def __init__(
        self,
        strategy: str,
        seed: int,
        end_day: date,
        tick_minutes: int = 15,
    ):
        valid_strategies = {
            "steady_daily",
            "maximize_constraints",
            "payload_first",
            "keep_final_assembly_fed",
        }

        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        if tick_minutes <= 0:
            raise ValueError("tick_minutes must be positive")

        self.strategy = strategy
        self.rng = random.Random(seed)

        self.start_dt = datetime(2026, 6, 1, 7, 0)

        if end_day < self.start_dt.date():
            raise ValueError("end_day cannot precede the simulation start")

        # Include the entire final calendar day.
        self.stop_dt = datetime.combine(
            end_day + timedelta(days=1),
            time.min,
        )

        self.duration = self.to_minutes(self.stop_dt)
        self.tick_minutes = tick_minutes

        self.env = simpy.Environment()

        self.routing = load_routing()
        self.bom = load_bom()
        self.resource_counts = load_resource_counts()
        self.mold_counts = load_molds()

        self.validate_inputs()

        # Containers represent interchangeable capacity units.
        # Zero-capacity pools are represented by None.
        self.resources = {
            name: (
                simpy.Container(self.env, capacity=count, init=count)
                if count > 0
                else None
            )
            for name, count in self.resource_counts.items()
        }

        self.molds = {
            name: (
                simpy.Container(self.env, capacity=count, init=count)
                if count > 0
                else None
            )
            for name, count in self.mold_counts.items()
        }

        self.inventory = Counter()
        self.in_system = Counter()
        self.shipped_by_month = Counter()
        self.utilization = Counter()
        self.event_log = []

        self.queue = deque()
        self.running = {}
        self.next_job_id = 0
        self.shipped_count = 0

        # Retained from the original model.
        # This is NOT a cap on total shipments.
        self.target_leaves = expand_required_leaves(
            "finished_product",
            220,
            self.bom,
        )

        self.launchable = [
            item for item in self.routing
            if item not in self.bom
        ]

        # Retain the original assembly order.
        self.parents = [
            "payload",
            "avionics",
            "tail",
            "wing_pair",
            "finished_product",
        ]

    def validate_inputs(self):
        for item, steps in self.routing.items():
            orders = [step.step_order for step in steps]

            if len(orders) != len(set(orders)):
                raise ValueError(f"Duplicate routing order for {item}")

            for step in steps:
                if min(
                    step.active_labor_min,
                    step.passive_wait_min,
                    step.people_required,
                    step.rework_min,
                ) < 0:
                    raise ValueError(
                        f"Negative routing value: {item}/{step.step_name}"
                    )

                if not 0 <= step.defect_rate <= 1:
                    raise ValueError(
                        f"Invalid defect rate: {item}/{step.step_name}"
                    )

                needs_resource = (
                    step.resource == "oven"
                    or (
                        step.active_labor_min > 0
                        and step.people_required > 0
                    )
                )

                if needs_resource:
                    if step.resource not in self.resource_counts:
                        raise ValueError(
                            f"Missing resource: {step.resource}"
                        )

                    units = (
                        max(1, step.people_required)
                        if step.resource == "oven"
                        else step.people_required
                    )

                    if units > self.resource_counts[step.resource]:
                        raise ValueError(
                            f"Step requires more capacity than available: "
                            f"{item}/{step.step_name}"
                        )

                if step.mold_occupied and step.mold_type:
                    if self.mold_counts[step.mold_type] <= 0:
                        raise ValueError(
                            f"No available molds for {step.mold_type}"
                        )

        for parent, children in self.bom.items():
            if parent not in self.routing:
                raise ValueError(f"Missing parent routing: {parent}")

            names = [child for child, _ in children]

            if len(names) != len(set(names)):
                raise ValueError(
                    f"Duplicate child rows in BOM for {parent}"
                )

            for child, quantity in children:
                if quantity <= 0:
                    raise ValueError(
                        f"Invalid BOM quantity: {parent}/{child}"
                    )

                if child not in self.routing:
                    raise ValueError(f"Missing child routing: {child}")

        # Detect circular BOM relationships.
        visiting = set()
        visited = set()

        def visit(item):
            if item in visiting:
                raise ValueError(f"Circular BOM involving {item}")

            if item in visited:
                return

            visiting.add(item)

            for child, _ in self.bom.get(item, []):
                visit(child)

            visiting.remove(item)
            visited.add(item)

        for item in self.bom:
            visit(item)

    def to_datetime(self, minutes=None):
        if minutes is None:
            minutes = self.env.now

        return self.start_dt + timedelta(minutes=minutes)

    def to_minutes(self, timestamp):
        return (timestamp - self.start_dt).total_seconds() / 60

    def work_window(self, timestamp):
        day = timestamp.date()
        shift_start = datetime.combine(day, time(7, 0))
        shift_end = shift_start + timedelta(
            minutes=working_minutes(day)
        )
        return shift_start, shift_end

    def is_working(self):
        now = self.to_datetime()
        start, end = self.work_window(now)
        return start <= now < end

    def next_work_start(self):
        now = self.to_datetime()
        day = now.date()

        while True:
            start = datetime.combine(day, time(7, 0))

            if working_minutes(day) > 0 and start > now:
                return self.to_minutes(start)

            day += timedelta(days=1)

    def log(self, event, item, job=None, **details):
        now = self.to_datetime()

        record = {
            "datetime": now.isoformat(sep=" "),
            "date": now.date().isoformat(),
            "month": now.strftime("%Y-%m"),
            "strategy": self.strategy,
            "event": event,
            "item": item,
            **details,
        }

        if job is not None:
            record["job_id"] = job.job_id

        self.event_log.append(record)

    def new_job(self, item):
        self.next_job_id += 1

        job = SimJob(
            job_id=self.next_job_id,
            item=item,
            ready_at=self.env.now,
        )

        self.queue.append(job)
        self.in_system[item] += 1
        return job

    def choose_launch_items(self):
        if self.strategy == "payload_first":
            return sorted(
                self.launchable,
                key=lambda item: (
                    not item.startswith("payload"),
                    item,
                ),
            )

        if self.strategy == "keep_final_assembly_fed":
            return sorted(
                self.launchable,
                key=lambda item: self.inventory[item],
            )

        return sorted(
            self.launchable,
            key=lambda item: (
                self.inventory[item]
                - self.target_leaves.get(item, 0)
            ),
        )

    def launch_cap(self, item):
        if self.strategy == "steady_daily":
            return 8

        if self.strategy == "payload_first":
            if item.startswith("payload"):
                return 40

        if self.strategy == "maximize_constraints":
            return 50

        return 20

    def create_assemblies_and_ship(self):
        for parent in self.parents:
            while (
                parent in self.routing
                and can_assemble(parent, self.inventory, self.bom)
            ):
                consume_children(parent, self.inventory, self.bom)
                job = self.new_job(parent)

                self.log(
                    "assembly_started",
                    parent,
                    job=job,
                    meaning="children committed; job queued",
                )

        month = self.to_datetime().strftime("%Y-%m")

        while self.inventory["finished_product"] > 0:
            self.inventory["finished_product"] -= 1
            self.shipped_by_month[month] += 1
            self.shipped_count += 1

            self.log(
                "product_shipped",
                "finished_product",
                product_number=self.shipped_count,
                month_shipped=self.shipped_by_month[month],
                cumulative_shipped=self.shipped_count,
            )

    def release_leaf_jobs(self):
        for item in self.choose_launch_items():
            if (
                self.inventory[item]
                < self.target_leaves.get(item, 0)
                and self.in_system[item] < self.launch_cap(item)
            ):
                job = self.new_job(item)

                self.log(
                    "item_launched",
                    item,
                    job=job,
                    in_system_after=self.in_system[item],
                )

    def start_queued_jobs(self):
        """Reserve available capacity and start eligible steps."""

        for job in list(self.queue):
            if job.ready_at > self.env.now:
                continue

            step = self.routing[job.item][job.step_index]

            needs_mold = (
                step.mold_occupied
                and bool(step.mold_type)
                and step.mold_type not in job.held_molds
            )

            if needs_mold:
                mold_pool = self.molds[step.mold_type]

                if mold_pool.level < 1:
                    continue

            is_oven = step.resource == "oven"

            units = (
                max(1, step.people_required)
                if is_oven
                else (
                    step.people_required
                    if step.active_labor_min > 0
                    else 0
                )
            )

            pool = self.resources.get(step.resource)

            if units and (pool is None or pool.level < units):
                continue

            # There is one dispatcher. Check-and-reserve occurs
            # without yielding between the checks and reservations.
            if needs_mold:
                self.molds[step.mold_type].get(1)
                job.held_molds.add(step.mold_type)

            if units:
                pool.get(units)

            self.queue.remove(job)
            self.running[job.job_id] = job

            self.env.process(
                self.run_step(job, step, units, is_oven)
            )

    def active_work(self, step, units, is_oven):
        """
        Perform active work only inside scheduled work hours.

        Non-oven workers are released at shift end and reacquired
        at the next working period.

        Oven capacity remains reserved for the entire step.
        """

        remaining = float(step.active_labor_min)
        pool = self.resources.get(step.resource)

        # Ordinary resource capacity was reserved by the dispatcher.
        held = units > 0 and not is_oven

        while remaining > 0:
            if not self.is_working():
                if held:
                    yield pool.put(units)
                    held = False

                yield self.env.timeout(
                    self.next_work_start() - self.env.now
                )
                continue

            if units > 0 and not is_oven and not held:
                yield pool.get(units)
                held = True

                # Capacity acquisition may have waited past shift end.
                if not self.is_working():
                    continue

            _, shift_end = self.work_window(self.to_datetime())

            segment = min(
                remaining,
                self.to_minutes(shift_end) - self.env.now,
            )

            # Credit actual elapsed active work, including partial
            # work if the run ends before this segment completes.
            segment_start = self.env.now
            yield self.env.timeout(segment)

            self.utilization[step.resource] += (
                self.env.now - segment_start
            ) * units

            remaining -= segment

            if remaining > 0:
                self.log(
                    "active_work_paused",
                    "",
                    resource=step.resource,
                    remaining_active_min=remaining,
                )

        if held:
            yield pool.put(units)

    def run_step(self, job, step, units, is_oven):
        self.log(
            "step_started",
            job.item,
            job=job,
            step_name=step.step_name,
            resource=step.resource,
            step_order=step.step_order,
            active_labor_min=step.active_labor_min,
            passive_wait_min=step.passive_wait_min,
            people_required=step.people_required,
            mold_type=step.mold_type,
        )

        if step.active_labor_min > 0:
            yield self.env.process(
                self.active_work(step, units, is_oven)
            )

        if step.passive_wait_min > 0:
            self.log(
                "passive_wait_started",
                job.item,
                job=job,
                step_name=step.step_name,
            )

            # Passive time continues overnight and on off days.
            yield self.env.timeout(step.passive_wait_min)

        if step.active_labor_min + step.passive_wait_min == 0:
            yield self.env.timeout(1)

        if is_oven:
            yield self.resources[step.resource].put(units)

        failed = (
            step.qa_required
            and self.rng.random() < step.defect_rate
        )

        if failed:
            self.log(
                "qa_failed_rework_started",
                job.item,
                job=job,
                step_name=step.step_name,
                resource=step.resource,
                rework_min=step.rework_min,
            )

            # Preserve the original semantics:
            # delay, then repeat the same complete step.
            job.ready_at = self.env.now + step.rework_min

        else:
            self.log(
                "step_finished",
                job.item,
                job=job,
                step_name=step.step_name,
                resource=step.resource,
                step_order=step.step_order,
            )

            job.step_index += 1
            job.ready_at = self.env.now

            # Release held molds only once the passed routing no
            # longer has a future occupied step needing them.
            remaining_steps = self.routing[job.item][job.step_index:]

            for mold_type in list(job.held_molds):
                still_needed = any(
                    future.mold_occupied
                    and future.mold_type == mold_type
                    for future in remaining_steps
                )

                if not still_needed:
                    yield self.molds[mold_type].put(1)
                    job.held_molds.remove(mold_type)

        self.running.pop(job.job_id)

        if job.step_index >= len(self.routing[job.item]):
            self.in_system[job.item] -= 1
            self.inventory[job.item] += 1

            self.log(
                "item_completed",
                job.item,
                job=job,
                inventory_after=self.inventory[job.item],
            )
        else:
            self.queue.append(job)

        # Dispatch immediately after a working-time completion,
        # rather than waiting for the next release tick.
        if self.is_working():
            self.create_assemblies_and_ship()
            self.start_queued_jobs()

    def dispatcher(self):
        """
        Keep the original release cadence: one eligible leaf job
        per item per 15-minute working tick.

        Completions themselves are event-driven.
        """

        while True:
            if not self.is_working():
                yield self.env.timeout(
                    self.next_work_start() - self.env.now
                )
                continue

            self.create_assemblies_and_ship()
            self.release_leaf_jobs()
            self.start_queued_jobs()

            _, shift_end = self.work_window(self.to_datetime())

            delay = min(
                self.tick_minutes,
                self.to_minutes(shift_end) - self.env.now,
            )

            yield self.env.timeout(delay)

    def results(self):
        demand = due_by_month()
        cumulative_demand = 0
        monthly = []

        for month, quantity in sorted(demand.items()):
            cumulative_demand += quantity

            # Include shipments in months omitted from demand.csv.
            cumulative_ship = sum(
                shipped
                for shipped_month, shipped
                in self.shipped_by_month.items()
                if shipped_month <= month
            )

            monthly.append(
                {
                    "month": month,
                    "shipped": self.shipped_by_month[month],
                    "cumulative_shipped": cumulative_ship,
                    "cumulative_demand": cumulative_demand,
                    "on_time": cumulative_ship >= cumulative_demand,
                }
            )

        return {
            "strategy": self.strategy,
            "total_shipped": self.shipped_count,
            "monthly": monthly,
            "ending_inventory": dict(self.inventory),
            "ending_wip": dict(+self.in_system),
            "utilization_minutes": dict(self.utilization),
            "event_log": self.event_log,
        }

    def run(self):
        self.env.process(self.dispatcher())
        self.env.run(until=self.duration)
        return self.results()


def simulate(
    strategy: str,
    seed: int = 7,
    end_day: date | None = None,
) -> dict[str, object]:
    factory = Factory(
        strategy=strategy,
        seed=seed,
        end_day=end_day or date(2027, 7, 31),
    )

    return factory.run()
