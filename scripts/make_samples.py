"""Generate small sample study-note PDFs into data/samples/ (DBMS, OS, CN).

Each page has a running header and a "Page N of M" footer so the cleaner's
header/footer removal is exercised. Replace/add your own notes for real use.

Usage: python scripts/make_samples.py [--out data/samples]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]

SAMPLES: dict[str, tuple[str, list[str]]] = {
    "DBMS_Notes.pdf": (
        "DBMS Lecture Notes - Unit 3 | B.Tech CSE",
        [
            """1. Introduction to Database Management Systems

A Database Management System (DBMS) is software that stores, organises and retrieves data while controlling concurrent access, security and recovery. Compared with a traditional file-processing system, a DBMS reduces data redundancy and inconsistency, enforces integrity constraints, supports concurrent users safely and provides backup and recovery.

Three-schema architecture: the external level (user views), the conceptual level (logical structure of the whole database) and the internal level (physical storage). The mappings between levels provide data independence.

Logical data independence is the ability to change the conceptual schema without changing external schemas or application programs. Physical data independence is the ability to change the internal schema (for example file organisation or indexes) without changing the conceptual schema.""",
            """2. Normalization

Normalization is the process of organising the attributes and tables of a relational database to minimise redundancy and to eliminate insertion, update and deletion anomalies. It works by decomposing a relation into smaller relations using functional dependencies, while keeping the decomposition lossless.

- Insertion anomaly: a fact cannot be stored without storing an unrelated fact.
- Update anomaly: the same fact is stored in many rows, so an update must change all of them.
- Deletion anomaly: deleting one fact unintentionally deletes another.

First Normal Form (1NF): every attribute holds only atomic (indivisible) values and there are no repeating groups.
Second Normal Form (2NF): the relation is in 1NF and no non-prime attribute is partially dependent on any candidate key (no partial dependency).
Third Normal Form (3NF): the relation is in 2NF and there is no transitive dependency of a non-prime attribute on a candidate key. Formally, for every functional dependency X -> A, either X is a superkey or A is a prime attribute.
Boyce-Codd Normal Form (BCNF): for every non-trivial functional dependency X -> A, X must be a superkey. BCNF is stricter than 3NF, and every BCNF relation is also in 3NF.""",
            """3. Functional Dependencies and Keys

A functional dependency X -> Y holds on a relation R if any two tuples that agree on the attributes X also agree on the attributes Y.

Armstrong's axioms are sound and complete inference rules for functional dependencies:
- Reflexivity: if Y is a subset of X, then X -> Y.
- Augmentation: if X -> Y, then XZ -> YZ.
- Transitivity: if X -> Y and Y -> Z, then X -> Z.

The closure of an attribute set X, written X+, is the set of all attributes functionally determined by X. If X+ contains every attribute of R, then X is a superkey.

A candidate key is a minimal superkey. The primary key is the candidate key chosen by the designer to identify tuples. A foreign key is an attribute set in one relation that refers to the primary key of another relation and enforces referential integrity.""",
            """4. Indexing and B-Trees

An index is an auxiliary structure that speeds up data retrieval at the cost of extra storage and slower writes.

A B-tree of order m is a self-balancing search tree in which every node has at most m children, every internal node except the root has at least ceil(m/2) children, and all leaves appear at the same level. Keys inside a node are kept in sorted order.

Advantages of B-trees:
- The tree always stays balanced, so search, insertion and deletion take O(log n) time.
- Each node holds many keys (high fan-out), so the tree is shallow and the number of disk I/O operations is small. This makes B-trees ideal for databases and file systems stored on disk.
- Nodes are kept at least half full, giving good storage utilisation.

A B+ tree stores all records (or record pointers) only in the leaf nodes, and the leaves are linked together. Internal nodes contain only keys used for routing. Because of the linked leaves, B+ trees are very efficient for range queries and sequential scans, which is why most relational databases use B+ trees for their indexes.""",
            """5. Transactions and Concurrency Control

A transaction is a logical unit of work that must satisfy the ACID properties:
- Atomicity: either all operations of the transaction happen or none do.
- Consistency: a transaction takes the database from one consistent state to another.
- Isolation: concurrent transactions do not see each other's intermediate results.
- Durability: once committed, changes survive system failures.

A schedule is serializable if its effect is equivalent to some serial execution of the same transactions. Conflict serializability can be tested with a precedence graph: the schedule is conflict serializable if and only if the graph has no cycle.

Two-Phase Locking (2PL) guarantees conflict serializability. In the growing phase a transaction may acquire locks but not release any; in the shrinking phase it may release locks but not acquire new ones. Strict 2PL holds all exclusive locks until commit, which also avoids cascading rollbacks. 2PL can still lead to deadlock, which is handled by detection (wait-for graph) or prevention (wait-die, wound-wait).""",
        ],
    ),
    "OS_Notes.pdf": (
        "Operating Systems - Class Notes (Semester 4)",
        [
            """1. Processes

A process is a program in execution. During its lifetime a process moves through the states new, ready, running, waiting (blocked) and terminated. The scheduler moves a process from ready to running; an I/O request moves it from running to waiting.

The Process Control Block (PCB) stores everything the OS needs to manage a process: process id, process state, program counter, CPU registers, scheduling information (priority), memory-management information (page tables), accounting information and the list of open files.

A context switch saves the state of the running process into its PCB and loads the saved state of the next process. Context switching is pure overhead because no useful work is done during the switch.""",
            """2. CPU Scheduling

Scheduling criteria: CPU utilisation, throughput, turnaround time, waiting time and response time.
Turnaround time = completion time - arrival time. Waiting time = turnaround time - burst time.

- First-Come, First-Served (FCFS): non-preemptive; simple, but suffers from the convoy effect when a long job delays short ones.
- Shortest Job First (SJF): gives the minimum average waiting time for a given set of processes, but requires knowing the next CPU burst length. Its preemptive version is Shortest Remaining Time First (SRTF).
- Round Robin (RR): each process gets a fixed time quantum and is then preempted and put at the end of the ready queue. A very large quantum makes RR behave like FCFS; a very small quantum causes too many context switches.
- Priority scheduling: the highest priority process runs first. Starvation of low-priority processes is solved by aging, which gradually increases the priority of waiting processes.""",
            """3. Deadlocks

A deadlock is a situation where a set of processes are blocked because each process holds a resource and waits for a resource held by another process in the set.

Coffman conditions - a deadlock can occur only if all four hold simultaneously:
1. Mutual exclusion: at least one resource is non-sharable.
2. Hold and wait: a process holds at least one resource while waiting for others.
3. No preemption: resources cannot be forcibly taken from a process.
4. Circular wait: a circular chain of processes exists, each waiting for a resource held by the next.

Deadlock handling strategies are prevention (break one of the four conditions), avoidance, detection with recovery, and ignoring the problem (the ostrich approach).

Banker's algorithm (deadlock avoidance): each process declares its maximum need in advance. A request is granted only if the resulting state is safe, meaning there exists a sequence in which every process can obtain its maximum need and finish. Need = Max - Allocation.""",
            """4. Memory Management and Paging

Paging divides physical memory into fixed-size frames and logical memory into pages of the same size. The page table maps each page number to a frame number, which removes external fragmentation (internal fragmentation can still occur in the last page).

A logical address is split into a page number p and a page offset d. The Translation Lookaside Buffer (TLB) is a small, fast associative cache of recent page-table entries that avoids a second memory access for most translations.

A page fault occurs when a referenced page is not in main memory; the OS loads it from disk, possibly replacing another page.

Page replacement algorithms:
- FIFO replaces the oldest page. FIFO suffers from Belady's anomaly: increasing the number of frames can increase the number of page faults.
- Optimal (OPT) replaces the page that will not be used for the longest time in the future. It gives the lowest fault rate but cannot be implemented in practice.
- Least Recently Used (LRU) replaces the page that has not been used for the longest time. LRU does not suffer from Belady's anomaly.""",
            """5. Process Synchronization

The critical-section problem requires a solution that satisfies three conditions:
- Mutual exclusion: only one process executes in its critical section at a time.
- Progress: if no process is in its critical section, the choice of the next process to enter cannot be postponed indefinitely.
- Bounded waiting: there is a limit on how many times other processes may enter their critical sections after a process has requested entry.

A semaphore is an integer variable accessed only through two atomic operations: wait (P), which decrements the value and blocks if it becomes negative, and signal (V), which increments the value and wakes a blocked process. A binary semaphore (mutex) takes only the values 0 and 1; a counting semaphore can take any non-negative value.

In the producer-consumer (bounded buffer) problem, a mutex protects the buffer, a semaphore 'empty' counts free slots and a semaphore 'full' counts filled slots.""",
        ],
    ),
    "CN_Notes.pdf": (
        "Computer Networks - Lecture Notes",
        [
            """1. Network Models

The OSI reference model has seven layers: Physical, Data Link, Network, Transport, Session, Presentation and Application. The TCP/IP model has four layers: Link, Internet, Transport and Application.

- The Data Link layer provides framing, error detection (for example CRC) and medium access control.
- The Network layer handles logical addressing and routing (IP).
- The Transport layer provides process-to-process delivery (TCP, UDP), using port numbers to identify applications.""",
            """2. Transmission Control Protocol

TCP is a connection-oriented, reliable, byte-stream transport protocol. It provides flow control using a sliding window and retransmits lost segments.

TCP three-way handshake (connection establishment):
1. The client sends a SYN segment with its initial sequence number x.
2. The server replies with SYN-ACK, carrying its own initial sequence number y and acknowledgement number x+1.
3. The client sends an ACK with acknowledgement number y+1. The connection is now established.

Connection termination uses a four-way exchange: FIN, ACK, FIN, ACK. The side that closes first enters the TIME_WAIT state for twice the maximum segment lifetime (2MSL).

TCP versus UDP: UDP is connectionless and unreliable but has lower overhead and latency, which suits DNS queries, streaming and online games. TCP guarantees ordered, reliable delivery and is used by HTTP, SMTP and FTP.""",
            """3. Routing Algorithms

Dijkstra's algorithm finds the shortest paths from a single source to all other nodes in a graph with non-negative edge weights. It is used by link-state routing protocols such as OSPF.

Steps: set the distance of the source to 0 and all others to infinity; repeatedly pick the unvisited node with the smallest tentative distance, mark it visited, and relax all its outgoing edges (if dist[u] + w(u,v) < dist[v], update dist[v]). Stop when all nodes are visited.

Time complexity of Dijkstra's algorithm: O(V^2) with a simple array, and O((V + E) log V) with a binary-heap priority queue. It does not work with negative edge weights.

The Bellman-Ford algorithm handles negative weights in O(V * E) time and is the basis of distance-vector routing protocols such as RIP. Distance-vector routing suffers from the count-to-infinity problem, which is reduced by split horizon and poison reverse.""",
            """4. IP Addressing and Subnetting

An IPv4 address is 32 bits long and is written in dotted-decimal notation, for example 192.168.1.10. An IPv6 address is 128 bits long.

CIDR notation writes the prefix length after a slash. A /24 network has 2^8 = 256 addresses, of which 254 are usable for hosts because the network address and the broadcast address are reserved. The subnet mask of a /24 network is 255.255.255.0.

Number of usable hosts in a subnet = 2^(32 - prefix) - 2.

Private IPv4 ranges are 10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16. Network Address Translation (NAT) lets many hosts with private addresses share one public address.""",
            """5. TCP Congestion Control

TCP congestion control adjusts the congestion window (cwnd) to avoid overloading the network.

- Slow start: cwnd starts at one maximum segment size and doubles every round-trip time (exponential growth) until it reaches the slow-start threshold (ssthresh).
- Congestion avoidance: after ssthresh, cwnd grows by one segment per round-trip time (additive increase).
- On a timeout, ssthresh is set to half of cwnd and cwnd is reset to one segment (multiplicative decrease).
- Fast retransmit: after three duplicate ACKs the sender retransmits the missing segment without waiting for the timeout. With fast recovery (TCP Reno), cwnd is halved instead of being reset to one.

This behaviour is known as AIMD: Additive Increase, Multiplicative Decrease.""",
        ],
    ),
}


def build_pdf(path: Path, header: str, pages: list[str]) -> None:
    """Write one sample PDF with header/footer on every page."""
    pdf = pymupdf.open()
    total = len(pages)
    for number, body in enumerate(pages, start=1):
        page = pdf.new_page(width=595, height=842)  # A4
        page.insert_text((50, 40), header, fontsize=9, fontname="helv")
        overflow = page.insert_textbox(pymupdf.Rect(50, 60, 545, 790), body, fontsize=10.5, fontname="helv")
        if overflow < 0:
            raise ValueError(f"{path.name} page {number} text does not fit")
        page.insert_text((270, 815), f"Page {number} of {total}", fontsize=9, fontname="helv")
    pdf.save(path)
    pdf.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "samples")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (header, pages) in SAMPLES.items():
        build_pdf(args.out / name, header, pages)
        print(f"wrote {args.out / name} ({len(pages)} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
