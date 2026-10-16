"""Deterministic simulated history + labeled routing requests.

Ground truth is a list of *agents*, each belonging to a *group* (= one real-world thread
the user cares about). Several agents can belong to one group: that is exactly how the
current system fragments context ("Email to Alice", "Alice reply", "Alice lease"...).

A routing decision is CORRECT when the request goes to any agent of the expected group
(or creates a new agent when the expected group is None, i.e. a genuinely new task).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

SEED = 7

FIRST = (
    "Alice Bob Carol Dave Erin Frank Grace Heidi Ivan Judy Kevin Laura Mallory Nick Olivia Peter Quinn Rita "
    "Sam Tina Uma Victor Wendy Xavier Yara Zane Aaron Bella Caleb Diana Ethan Fiona George Hannah Isaac Julia "
    "Karl Lena Mason Nora Oscar Priya Quentin Rosa Simon Talia Umar Vera Walter Ximena Yusuf Zoe Adam Beth "
    "Chris Dina Evan Faye Gus Hana Igor Jade Kyle Lola Milo Nina Owen Pia Raj Sara Theo Una Vlad Willa "
    "Xena Yuri Zara Abel Bea Cody Dora Eli Flora Gil Hugo Iris Joel Kira Leo Maya Ned Opal Pablo Reed "
    "Sage Tess Uri Vince Wes Yael Zeke Ann"
).split()
LAST = (
    "Chen Patel Garcia Kim Nguyen Singh Lopez Brown Davis Wilson Moore Clark Lewis Young Hall Allen Wright "
    "Scott Green Baker Adams Nelson Hill Ramirez Campbell Mitchell Roberts Carter Phillips Evans Turner Torres "
    "Parker Collins Edwards Stewart Flores Morris Murphy Rivera Cook Rogers Morgan Peterson Cooper Reed Bailey "
    "Bell Gomez Kelly Howard Ward Cox Diaz Richardson Wood Watson Brooks Bennett Gray James Reyes Cruz Hughes "
    "Price Myers Long Foster Sanders Ross Morales Powell Sullivan Russell Ortiz Jenkins Gutierrez Perry "
    "Butler Barnes Fisher Henderson Coleman Simmons Patterson Jordan Reynolds Hamilton Graham Kim Gonzales "
    "Alexander Ramos Wallace Griffin West Cole Hayes Chavez Gibson Bryant Ellis Stevens Murray Ford Marshall"
).split()
PERSON_TOPICS = (
    "lease renewal;invoice 2041;conference talk;wedding RSVP;contract review;apartment viewing;"
    "bike repair quote;visa documents;tax receipts;project handoff;birthday dinner;car insurance;"
    "dentist reschedule;book club;moving boxes;portfolio feedback;reference letter;gym membership;"
    "photo shoot;catering order;thesis meeting;concert tickets;roommate deposit;plumber visit;"
    "freelance proposal;NDA signature;podcast guest;school pickup;hackathon team;vendor pricing"
).split(";")
TOPIC_ONLY = (
    "Vercel job offer;Tax filing 2025;Flight to Lisbon;Passport renewal;Health insurance claim;"
    "Kitchen remodel;Conference travel;Laptop warranty;Car registration;Newsletter draft;"
    "Weekly report;Gym schedule;Grocery order;Roof inspection;Domain renewal;Mortgage refinance;"
    "Thesis submission;Charity donation;Wedding gift;Garden irrigation;Tokyo itinerary;Resume update;"
    "Phone plan switch;Storage unit;Jury duty notice;Library fines;Pet vaccination;Solar quote;"
    "Credit card dispute;Internship applications;Meal prep plan;Furniture return;Warranty claim;"
    "Marathon training;Bank statement review;Utility bills;Pottery class;Camping trip;Board game night;"
    "Parking permit;Visa appointment;Standing desk order;Photo backup;Language course;Open source PR;"
    "Insurance renewal;Subscription audit;Yearbook order;Recycling pickup;Flu shot;Co-working pass;"
    "Garage sale;Alumni dinner;Budget review;Bike registration;Carpool schedule;Plant watering;"
    "Rent increase;Mailing address change;Recipe collection;Drone license;Estate paperwork;Tax refund"
).split(";")


@dataclass
class Agent:
    name: str
    group: str
    kind: str  # person | topic
    description: str
    people: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


@dataclass
class Request:
    id: str
    category: str  # exact | indirect | dup_trap | new
    text: str
    expected_group: Optional[str]  # None => should create a NEW agent


def _variants(rng: random.Random, first: str, last: str, topic: str) -> List[str]:
    pool = [
        f"Email to {first}",
        f"{first} reply",
        f"{first} follow-up",
        f"Follow up with {first} {last}",
        f"{first} {topic}",
        f"{topic.title()} - {first}",
        f"Message {first} {last}",
    ]
    rng.shuffle(pool)
    n = rng.choices([1, 2, 3, 4], weights=[55, 22, 13, 10])[0]
    return pool[:n]


def build_agents(target: int = 300) -> List[Agent]:
    rng = random.Random(SEED)
    firsts = FIRST[:]
    rng.shuffle(firsts)
    agents: List[Agent] = []
    gid = 0
    seen_names = set()

    def add(name: str, **kw) -> None:
        if name in seen_names:
            return
        seen_names.add(name)
        agents.append(Agent(name=name, **kw))

    person_groups = 0
    for first in firsts:
        if len(agents) >= target * 0.62:
            break
        last = rng.choice(LAST)
        topic = rng.choice(PERSON_TOPICS)
        gid += 1
        group = f"g{gid:03d}"
        person_groups += 1
        for name in _variants(rng, first, last, topic):
            add(name, group=group, kind="person",
                description=f"Emails and coordination with {first} {last} about {topic}.",
                people=[f"{first} {last}"], topics=[topic])
    for topic in TOPIC_ONLY:
        if len(agents) >= target:
            break
        gid += 1
        group = f"g{gid:03d}"
        variants = [topic, f"{topic} follow-up", f"{topic} research", f"Check {topic.lower()}"]
        rng.shuffle(variants)
        for name in variants[: rng.choices([1, 2, 3], weights=[60, 28, 12])[0]]:
            add(name, group=group, kind="topic",
                description=f"Ongoing task: {topic}.", topics=[topic])
    return agents


def build_requests(agents: List[Agent], n_per_category: int = 10) -> List[Request]:
    rng = random.Random(SEED + 1)
    by_group: Dict[str, List[Agent]] = {}
    for a in agents:
        by_group.setdefault(a.group, []).append(a)

    person_groups = [g for g, v in by_group.items() if v[0].kind == "person"]
    multi = [g for g in person_groups if len(by_group[g]) >= 3]
    rng.shuffle(person_groups)
    rng.shuffle(multi)
    topic_groups = [g for g, v in by_group.items() if v[0].kind == "topic"]
    rng.shuffle(topic_groups)

    reqs: List[Request] = []

    def person(g: str) -> Agent:
        return by_group[g][0]

    # exact: names the person explicitly
    for i, g in enumerate(person_groups[:n_per_category]):
        a = person(g)
        who = a.people[0]
        topic = a.topics[0]
        text = rng.choice([
            f"email {who} and ask for an update on the {topic}",
            f"send {who.split()[0]} a quick note about the {topic}",
            f"can you follow up with {who} regarding the {topic}?",
        ])
        reqs.append(Request(f"exact-{i+1:02d}", "exact", text, g))

    # indirect: refers to the thread without using the agent's name. Person threads are referred
    # to by LAST name + topic (last name unique in the history, so the label is unambiguous);
    # topic threads by their topic. Name-only routing has to guess; descriptions make it easy.
    used = set(person_groups[:n_per_category])
    last_counts: Dict[str, int] = {}
    for g in person_groups:
        last = person(g).people[0].split()[-1]
        last_counts[last] = last_counts.get(last, 0) + 1
    unique_last = [g for g in person_groups if g not in used and last_counts[person(g).people[0].split()[-1]] == 1]
    indirect_pool = unique_last[: n_per_category // 2]
    indirect_pool += topic_groups[: n_per_category - len(indirect_pool)]
    for i, g in enumerate(indirect_pool):
        a = person(g)
        topic = a.topics[0]
        text = rng.choice([  # (draw kept so later categories' random stream is unchanged)
            f"any news on the {topic}? check and let me know",
            f"remind me where things stand with the {topic}",
            f"can you push the {topic} forward today",
        ])
        if a.kind == "person":
            last = a.people[0].split()[-1]
            text = rng.choice([
                f"any word from {last} on the {topic}? check and let me know",
                f"where do things stand with {last} about the {topic}?",
            ])
        reqs.append(Request(f"indirect-{i+1:02d}", "indirect", text, g))

    # dup_trap: group already fragmented across 3-4 agents
    for i, g in enumerate(multi[:n_per_category]):
        a = person(g)
        who = a.people[0]
        topic = a.topics[0]
        text = rng.choice([
            f"also tell {who.split()[0]} the {topic} is confirmed on my side",
            f"write {who} again about the {topic}, short and polite",
        ])
        reqs.append(Request(f"duptrap-{i+1:02d}", "dup_trap", text, g))

    # new: people/topics that exist nowhere in the history
    new_people = [("Marguerite Okonkwo", "kayak rental"), ("Tobias Lindqvist", "sublet agreement"),
                  ("Anneliese Bauer", "violin lessons"), ("Declan Ferreira", "warehouse booking"),
                  ("Sunita Rao", "art commission"), ("Bjorn Halvorsen", "boat storage"),
                  ("Imani Washington", "catering tasting"), ("Lucía Navarro", "translation job"),
                  ("Kofi Mensah", "farm visit"), ("Hiroshi Tanabe", "ramen pop-up")]
    for i, (who, topic) in enumerate(new_people[:n_per_category]):
        reqs.append(Request(f"new-{i+1:02d}", "new",
                            f"email {who} about the {topic}, I haven't spoken to them before", None))
    return reqs


# Requests that refer to a thread WITHOUT any word used in the agent's name or description (synonyms and
# paraphrases). The main dataset shares surface words with the agents, so a word ranker looks perfect on it;
# this set is what separates word matching from semantic (embedding) matching. Kept out of ground_truth() so
# the baseline dataset (and its fingerprint) is unchanged. Only topic-only threads are used: their topic is
# unique, so each label is unambiguous.
PARAPHRASES = {
    "Passport renewal": "can you check on getting my travel document renewed",
    "Health insurance claim": "any progress on the medical coverage reimbursement?",
    "Kitchen remodel": "how is the cooking area renovation going",
    "Laptop warranty": "follow up on the computer guarantee coverage",
    "Car registration": "sort out my vehicle licence for this year",
    "Mortgage refinance": "where are we on restructuring my home loan",
    "Roof inspection": "did anyone confirm the shingles checkup",
    "Flight to Lisbon": "look into the airline ticket to Portugal",
    "Solar quote": "get an update on the photovoltaic panel estimate",
    "Jury duty notice": "what do I need to do about the court summons",
    "Pet vaccination": "book the shots for my dog",
    "Marathon training": "plan the long distance race preparation this week",
    "Standing desk order": "has the adjustable workstation purchase shipped",
    "Storage unit": "ask about the warehouse locker rental",
}


def build_paraphrase_requests(agents: List[Agent]) -> List[Request]:
    group_of_topic = {}
    for a in agents:
        if a.kind == "topic":
            group_of_topic[a.topics[0]] = a.group
    return [
        Request(f"para-{i+1:02d}", "paraphrase", text, group_of_topic[topic])
        for i, (topic, text) in enumerate(PARAPHRASES.items())
    ]


def first_request(agent) -> str:
    """A realistic first request for this agent's own name. Production builds an agent's description (and its
    people) from its first request, so evals seed agents this way - not with the group-level `description`,
    which is identical within a thread and makes retrieval and duplicate detection look easier than they are."""
    name, topic = agent.name, agent.topics[0]
    if agent.kind == "topic":
        if name.endswith("follow-up"):
            return f"follow up on the {topic.lower()}"
        if name.endswith("research"):
            return f"research options for the {topic.lower()}"
        if name.startswith("Check"):
            return f"check the status of the {topic.lower()}"
        return f"take care of the {topic.lower()}"
    full = agent.people[0]
    first = full.split()[0]
    if name == f"Email to {first}":
        return f"email {first} about the {topic}"
    if name == f"{first} reply":
        return f"reply to {first}'s last message"
    if name == f"{first} follow-up":
        return f"follow up with {first} if there is no answer by Friday"
    if name == f"Follow up with {full}":
        return f"follow up with {full} on the {topic}"
    if name == f"Message {full}":
        return f"send {full} a message about the {topic}"
    if name.startswith(first):
        return f"handle the {topic} with {first}"
    return f"{topic}: coordinate with {first}"


def ground_truth() -> Dict:
    agents = build_agents()
    return {
        "agents": [asdict(a) for a in agents],
        "requests": [asdict(r) for r in build_requests(agents)],
    }


if __name__ == "__main__":
    gt = ground_truth()
    groups = {a["group"] for a in gt["agents"]}
    print(f"{len(gt['agents'])} agents in {len(groups)} groups, {len(gt['requests'])} requests")
    for r in gt["requests"][:3] + gt["requests"][-2:]:
        print(r)
