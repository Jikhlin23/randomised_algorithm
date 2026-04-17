#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdint>
#include <iomanip>
#include <string>
#include <algorithm>
#include <fstream>

// ---------------------------------------------------------------------------
// HASH FUNCTION FAMILIES
//
// We need hash functions that are fast to compute and spread keys uniformly.
// A simple multiplicative hash (multiply by an odd constant, then XOR-fold)
// works well in practice for 32-bit keys.
// For Cuckoo Hashing specifically, we XOR three independent multiplicative
// hashes together to get a stronger, less predictable function — this is
// important because Cuckoo is more sensitive to hash quality than the
// other schemes.
// ---------------------------------------------------------------------------

class MultiplicativeHash {
    uint32_t a;
public:
    MultiplicativeHash(uint32_t multiplier) : a(multiplier) {
        // Hash multipliers must be odd for the math to work out nicely
        if (this->a % 2 == 0) this->a += 1;
    }
    inline uint32_t operator()(uint32_t x) const {
        uint32_t h = a * x;
        // XOR with a right-shifted version to mix the high and low bits
        return h ^ (h >> 16);
    }
};

class CuckooStrongHash {
    MultiplicativeHash h1, h2, h3;
public:
    CuckooStrongHash(uint32_t a1, uint32_t a2, uint32_t a3)
        : h1(a1), h2(a2), h3(a3) {}
    inline uint32_t operator()(uint32_t x) const {
        // Combining three independent hashes via XOR gives much better
        // uniformity than a single hash — Cuckoo Hashing requires this
        // to avoid insertion failures caused by hash collisions
        return h1(x) ^ h2(x) ^ h3(x);
    }
};

// ---------------------------------------------------------------------------
// HASH TABLE BASE CLASS
//
// All six data structures (Linear Probing, Padded Linear Probing,
// Double Hashing, Two-Way Chaining, Cuckoo Symmetric, Cuckoo Asymmetric)
// share this common interface. The benchmark engine only needs to call
// insert() and lookup() without knowing which scheme it is talking to.
//
// insert() returns the number of probes/kicks taken, which is the main
// metric we are measuring. It returns -1 if the insertion failed
// (e.g. table full, or Cuckoo detected a cycle).
// ---------------------------------------------------------------------------

class HashTable {
protected:
    size_t size;
    size_t count = 0;
public:
    HashTable(size_t s) : size(s) {}
    virtual ~HashTable() = default;

    virtual int insert(uint32_t key) = 0;
    virtual bool lookup(uint32_t key) const = 0;
    virtual std::string name() const = 0;
    virtual void reset() { count = 0; }
};

// ---------------------------------------------------------------------------
// 1. LINEAR PROBING (STANDARD)
//
// The simplest open-addressing scheme. Every key is stored directly in the
// table array at its hash position, or at the next empty slot scanning
// forward. Because consecutive slots are adjacent in memory, the CPU
// prefetcher can load several candidate slots in a single cache line fetch,
// giving Linear Probing excellent cache behaviour at low load factors.
//
// The cost, however, is "primary clustering": long runs of occupied slots
// build up over time, forcing new insertions to scan further and further.
// At high load factors this causes the probe count to explode.
//
// Probe count starts at 1 because checking the initial hash position is
// itself a memory access — it would be wrong to not count it.
// ---------------------------------------------------------------------------

class LinearProbing : public HashTable {
    struct Slot {
        uint32_t key;
        bool occupied = false;
    };
    std::vector<Slot> table;
    MultiplicativeHash hasher;

public:
    LinearProbing(size_t s, uint32_t seed) : HashTable(s), table(s), hasher(seed) {}
    std::string name() const override { return "Linear_Probing_Standard"; }

    int insert(uint32_t key) override {
        if (count >= size) return -1;
        size_t idx = hasher(key) % size;
        int probes = 1;
        while (table[idx].occupied) {
            if (table[idx].key == key) return probes;
            idx = (idx + 1) % size;
            probes++;
        }
        table[idx] = {key, true};
        count++;
        return probes;
    }

    bool lookup(uint32_t key) const override {
        size_t idx = hasher(key) % size;
        size_t start = idx;
        while (table[idx].occupied) {
            if (table[idx].key == key) return true;
            idx = (idx + 1) % size;
            if (idx == start) break;
        }
        return false;
    }

    void reset() override {
        count = 0;
        std::fill(table.begin(), table.end(), Slot{});
    }
};

// ---------------------------------------------------------------------------
// 2. LINEAR PROBING (PADDED — 64 bytes per slot)
//
// This is a deliberately handicapped version of Linear Probing designed to
// isolate the algorithmic cost from the cache benefit.
//
// In the standard version, 16 slots fit in one 64-byte cache line, so a
// single cache miss often resolves several consecutive probes for free.
// Here, each slot is padded to exactly 64 bytes (one full cache line) using
// alignas(64). This forces every single probe to generate its own cache miss,
// making the hardware pay for each comparison individually.
//
// By comparing Standard vs Padded, we can see exactly how much of Linear
// Probing's performance advantage comes from the cache rather than the
// algorithm itself.
// ---------------------------------------------------------------------------

class LinearProbingPadded : public HashTable {
    struct alignas(64) Slot {
        uint32_t key       = 0;
        bool     occupied  = false;
        char     _pad[59]  = {};
    };
    std::vector<Slot> table;
    MultiplicativeHash hasher;

public:
    LinearProbingPadded(size_t s, uint32_t seed)
        : HashTable(s), table(s), hasher(seed) {}
    std::string name() const override { return "Linear_Probing_PADDED"; }

    int insert(uint32_t key) override {
        if (count >= size) return -1;
        size_t idx = hasher(key) % size;
        int probes = 1;
        while (table[idx].occupied) {
            if (table[idx].key == key) return probes;
            idx = (idx + 1) % size;
            probes++;
        }
        table[idx].key      = key;
        table[idx].occupied = true;
        count++;
        return probes;
    }

    bool lookup(uint32_t key) const override {
        size_t idx   = hasher(key) % size;
        size_t start = idx;
        while (table[idx].occupied) {
            if (table[idx].key == key) return true;
            idx = (idx + 1) % size;
            if (idx == start) break;
        }
        return false;
    }

    void reset() override {
        count = 0;
        std::fill(table.begin(), table.end(), Slot{});
    }
};

// ---------------------------------------------------------------------------
// 3. DOUBLE HASHING
//
// An improvement over Linear Probing that avoids primary clustering by using
// a second hash function to determine the step size between probes. Because
// each key gets its own unique step size, colliding keys spread out across
// the table rather than bunching together.
//
// The expected probe count for an insertion at load factor α is 1/(1−α),
// which is strictly better than Linear Probing. The trade-off is that
// probes jump to non-consecutive memory addresses, losing the cache
// prefetching benefit that Linear Probing enjoys.
//
// Step size must be odd (or coprime with table size) to guarantee that
// the probe sequence visits every slot before repeating. Adding 1 after
// the modulo ensures step ≥ 1, preventing an infinite loop at step = 0.
// ---------------------------------------------------------------------------

class DoubleHashing : public HashTable {
    struct Slot {
        uint32_t key;
        bool occupied = false;
    };
    std::vector<Slot> table;
    MultiplicativeHash h1, h2;

public:
    DoubleHashing(size_t s, uint32_t s1, uint32_t s2)
        : HashTable(s), table(s), h1(s1), h2(s2) {}
    std::string name() const override { return "Double_Hashing"; }

    int insert(uint32_t key) override {
        if (count >= size) return -1;
        size_t idx  = h1(key) % size;
        size_t step = (h2(key) % (size - 1)) + 1;
        int probes  = 1;
        while (table[idx].occupied) {
            if (table[idx].key == key) return probes;
            idx = (idx + step) % size;
            probes++;
            if (probes > (int)size) return -1;
        }
        table[idx] = {key, true};
        count++;
        return probes;
    }

    bool lookup(uint32_t key) const override {
        size_t idx   = h1(key) % size;
        size_t step  = (h2(key) % (size - 1)) + 1;
        size_t start = idx;
        while (table[idx].occupied) {
            if (table[idx].key == key) return true;
            idx = (idx + step) % size;
            if (idx == start) break;
        }
        return false;
    }

    void reset() override {
        count = 0;
        std::fill(table.begin(), table.end(), Slot{});
    }
};

// ---------------------------------------------------------------------------
// 4. TWO-WAY CHAINING  ("Power of Two Choices")
//
// Instead of a single hash table, we maintain one table of linked lists
// (buckets). Each key is hashed to two candidate buckets. On insertion,
// we check the length of both buckets and place the key in whichever is
// shorter. This "power of two choices" idea dramatically reduces the
// maximum chain length from O(log n / log log n) to O(log log n).
//
// The probe count we record is: 2 (one header check per bucket to read
// their lengths) plus the number of elements in the chosen bucket before
// the new key is added. This accurately models the cost of scanning the
// chain to find the insertion point.
// ---------------------------------------------------------------------------

class TwoWaySymmetric : public HashTable {
    std::vector<std::vector<uint32_t>> table;
    MultiplicativeHash h1, h2;

public:
    TwoWaySymmetric(size_t s, uint32_t s1, uint32_t s2)
        : HashTable(s), table(s), h1(s1), h2(s2) {}
    std::string name() const override { return "TwoWay_Chaining"; }

    int insert(uint32_t key) override {
        size_t i1 = h1(key) % size;
        size_t i2 = h2(key) % size;

        // Make sure we are not inserting a duplicate
        for (auto k : table[i1]) if (k == key) return 0;
        for (auto k : table[i2]) if (k == key) return 0;

        size_t len1 = table[i1].size();
        size_t len2 = table[i2].size();
        int probes;
        if (len1 <= len2) {
            probes = 2 + (int)len1;
            table[i1].push_back(key);
        } else {
            probes = 2 + (int)len2;
            table[i2].push_back(key);
        }
        count++;
        return probes;
    }

    bool lookup(uint32_t key) const override {
        size_t i1 = h1(key) % size;
        for (auto k : table[i1]) if (k == key) return true;
        size_t i2 = h2(key) % size;
        for (auto k : table[i2]) if (k == key) return true;
        return false;
    }

    void reset() override {
        count = 0;
        for (auto& bucket : table) bucket.clear();
    }
};

// ---------------------------------------------------------------------------
// 5. CUCKOO HASHING — SYMMETRIC (equal table sizes, 1:1 ratio)
//
// The main algorithm we are studying. Two separate tables T1 and T2, each
// of size TABLE_SIZE/2, and two independent hash functions. Every key lives
// in exactly one of its two candidate cells — this is what makes lookup
// always cost exactly 2 memory accesses, no matter how full the tables are.
//
// Insertion works by "kicking out" the current occupant of the target cell
// and sending it to its alternative location, repeating until a vacant cell
// is found. The kick count we record is the length of this displacement
// chain, which is the insertion cost equivalent to "probes" in other schemes.
//
// If the chain runs for MAX_LOOPS steps without finding a vacant cell, the
// current hash functions have created an unresolvable cycle — the caller
// should pick new hash functions and rehash. We use MAX_LOOPS = 512, which
// is well above the theoretical value (~60 for our table size) to avoid
// falsely triggering rehashes on unlucky but valid insertions.
//
// Load factor note: the total capacity is TABLE_SIZE, split evenly as
// TABLE_SIZE/2 per table. Inserting TABLE_SIZE*lf keys means each table
// is at load lf — this is the correct way to account for Cuckoo's memory.
// ---------------------------------------------------------------------------

class CuckooSymmetric : public HashTable {
    struct Slot {
        uint32_t key;
        bool occupied = false;
    };
    std::vector<Slot> t1, t2;
    size_t half_size;
    CuckooStrongHash hasher1, hasher2;
    const size_t MAX_LOOPS = 512;

public:
    CuckooSymmetric(size_t total_size, CuckooStrongHash h1, CuckooStrongHash h2)
        : HashTable(total_size),
          half_size(total_size / 2),
          t1(total_size / 2),
          t2(total_size / 2),
          hasher1(h1), hasher2(h2) {}

    std::string name() const override { return "Cuckoo_Symmetric"; }

    int insert(uint32_t key) override {
        if (lookup(key)) return 0;
        uint32_t curr = key;
        int kicks = 0;
        bool in_t1 = true;

        // Always start by trying to place the key in T1
        size_t pos = hasher1(curr) % half_size;

        for (size_t i = 0; i < MAX_LOOPS; ++i) {
            if (in_t1) {
                if (!t1[pos].occupied) {
                    t1[pos] = {curr, true};
                    count++;
                    return kicks;
                }
                // Cell is occupied — evict the current resident and
                // send it to its alternative location in T2
                std::swap(curr, t1[pos].key);
                in_t1 = false;
                pos = hasher2(curr) % half_size;
            } else {
                if (!t2[pos].occupied) {
                    t2[pos] = {curr, true};
                    count++;
                    return kicks;
                }
                // Same eviction logic but now moving back to T1
                std::swap(curr, t2[pos].key);
                in_t1 = true;
                pos = hasher1(curr) % half_size;
            }
            kicks++;
        }
        return -1;
    }

    bool lookup(uint32_t key) const override {
        if (t1[hasher1(key) % half_size].occupied &&
            t1[hasher1(key) % half_size].key == key) return true;
        if (t2[hasher2(key) % half_size].occupied &&
            t2[hasher2(key) % half_size].key == key) return true;
        return false;
    }

    void reset() override {
        count = 0;
        std::fill(t1.begin(), t1.end(), Slot{});
        std::fill(t2.begin(), t2.end(), Slot{});
    }
};

// ---------------------------------------------------------------------------
// 6. CUCKOO HASHING — ASYMMETRIC (2:1 ratio, T1 is twice the size of T2)
//
// A variant of Cuckoo Hashing where the two tables are deliberately unequal.
// T1 gets 2/3 of the total memory and T2 gets 1/3. Because T1 is larger, it
// has more empty cells at any given load factor, so the insertion procedure
// finds a free slot in T1 more often. This pushes roughly 76% of keys into
// T1 (versus ~63% in the symmetric scheme).
//
// Why does that matter? The lookup procedure always checks T1 first. If the
// key is found there, the second memory access to T2 is skipped entirely.
// So more keys in T1 means more successful lookups that cost only 1 memory
// access instead of 2, improving the average lookup speed.
//
// The total memory used is identical to the symmetric scheme — this is a
// free performance improvement that any production implementation should use.
// ---------------------------------------------------------------------------

class CuckooAsymmetric : public HashTable {
    struct Slot {
        uint32_t key;
        bool occupied = false;
    };
    std::vector<Slot> t1, t2;
    size_t s1, s2;
    CuckooStrongHash hasher1, hasher2;
    const size_t MAX_LOOPS = 512;

public:
    CuckooAsymmetric(size_t total_size, CuckooStrongHash h1, CuckooStrongHash h2)
        : HashTable(total_size), hasher1(h1), hasher2(h2) {
        s1 = (total_size * 2) / 3;
        s2 = total_size - s1;
        t1.resize(s1);
        t2.resize(s2);
    }

    std::string name() const override { return "Cuckoo_Asymmetric"; }

    int insert(uint32_t key) override {
        if (lookup(key)) return 0;
        uint32_t curr = key;
        int kicks = 0;
        bool in_t1 = true;
        size_t pos = hasher1(curr) % s1;

        for (size_t i = 0; i < MAX_LOOPS; ++i) {
            if (in_t1) {
                if (!t1[pos].occupied) {
                    t1[pos] = {curr, true};
                    count++;
                    return kicks;
                }
                std::swap(curr, t1[pos].key);
                in_t1 = false;
                pos = hasher2(curr) % s2;
            } else {
                if (!t2[pos].occupied) {
                    t2[pos] = {curr, true};
                    count++;
                    return kicks;
                }
                std::swap(curr, t2[pos].key);
                in_t1 = true;
                pos = hasher1(curr) % s1;
            }
            kicks++;
        }
        return -1;
    }

    bool lookup(uint32_t key) const override {
        if (t1[hasher1(key) % s1].occupied &&
            t1[hasher1(key) % s1].key == key) return true;
        if (t2[hasher2(key) % s2].occupied &&
            t2[hasher2(key) % s2].key == key) return true;
        return false;
    }

    void reset() override {
        count = 0;
        std::fill(t1.begin(), t1.end(), Slot{});
        std::fill(t2.begin(), t2.end(), Slot{});
    }
};

// ---------------------------------------------------------------------------
// BENCHMARKING ENGINE
//
// This is the main experiment loop. For each algorithm and each load factor
// point, we run NUM_TRIALS independent trials. In each trial we:
//   1. Create a fresh hash table with newly randomised hash functions
//   2. Insert num_keys keys and record the total time and probe count
//   3. Immediately look up all the same keys and record the lookup time
//
// Using fresh hash functions per trial is important — it prevents any single
// lucky or unlucky hash function from biasing the average, and it models
// real-world usage where you cannot predict the key distribution in advance.
//
// If a trial fails (insert returns -1), we skip it and continue with the
// next trial rather than aborting the entire load factor point. This is the
// correct behaviour for Cuckoo Hashing, where a cycle detection failure with
// one set of hash functions is an expected rare event, not a fatal error.
// Averages are computed only over the trials that succeeded.
//
// Two output files are written:
//   academic_benchmark_AVERAGES.csv — one row per (algorithm, load factor)
//   academic_benchmark_RAW.csv      — one row per individual trial
// ---------------------------------------------------------------------------

void run_benchmark() {
    const size_t TABLE_SIZE = 1 << 20;
    const size_t NUM_TRIALS = 5000;

    std::cout << "\n======================================================\n";
    std::cout << "[SYSTEM] Starting Benchmark Suite...\n";
    std::cout << "[SYSTEM] Table Size : " << TABLE_SIZE << " slots\n";
    std::cout << "[SYSTEM] Trials/point: " << NUM_TRIALS << "\n";
    std::cout << "======================================================\n\n";

    std::ofstream avg_file("academic_benchmark_AVERAGES.csv");
    std::ofstream raw_file("academic_benchmark_RAW.csv");

    if (!avg_file.is_open() || !raw_file.is_open()) {
        std::cerr << "[ERROR] Could not create CSV files.\n";
        return;
    }

    avg_file << "Algorithm,LoadFactor,AvgInsertTime(ns),AvgLookupTime(ns),"
                "AvgInsertProbesOrKicks,Status\n";
    raw_file << "Algorithm,LoadFactor,TrialNumber,InsertTime(ns),"
                "LookupTime(ns),InsertProbesOrKicks,Status\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist;

    // Generate all keys up front using a fixed seed so every algorithm
    // sees exactly the same sequence of keys — this keeps the comparison fair
    size_t max_keys = static_cast<size_t>(TABLE_SIZE * 0.99);
    std::vector<uint32_t> dataset(max_keys);
    for (size_t i = 0; i < max_keys; ++i)
        dataset[i] = dist(rng);

    // Open addressing can be tested all the way to α = 0.98.
    // Cuckoo Hashing has a hard theoretical limit of α < 0.5 per table,
    // so we only test it up to 0.49 to stay within the valid operating range.
    std::vector<double> load_factors_open, load_factors_cuckoo;
    for (double lf = 0.02; lf <= 0.98; lf += 0.02)
        load_factors_open.push_back(lf);
    for (double lf = 0.02; lf <= 0.49; lf += 0.02)
        load_factors_cuckoo.push_back(lf);

    struct AlgoConfig {
        std::string name;
        bool is_cuckoo;
    };
    std::vector<AlgoConfig> algos = {
        {"Linear_Probing_Standard", false},
        {"Linear_Probing_PADDED",   false},
        {"Double_Hashing",          false},
        {"TwoWay_Chaining",         false},
        {"Cuckoo_Symmetric",        true },
        {"Cuckoo_Asymmetric",       true },
    };

    for (const auto& algo : algos) {
        const std::string& name = algo.name;
        const std::vector<double>& lf_range =
            algo.is_cuckoo ? load_factors_cuckoo : load_factors_open;

        std::cout << ">> Benchmarking: " << name << "\n";

        for (double lf : lf_range) {
            std::cout << "   -> LF=" << std::fixed << std::setprecision(2)
                      << lf << " ... " << std::flush;

            size_t num_keys     = static_cast<size_t>(TABLE_SIZE * lf);
            double total_insert = 0, total_lookup = 0, total_probes = 0;
            size_t successful_trials = 0;

            for (size_t t = 0; t < NUM_TRIALS; ++t) {

                // Fresh random seeds for every trial so each trial is
                // an independent experiment with its own hash functions
                uint32_t seed1 = dist(rng), seed2 = dist(rng);
                CuckooStrongHash strong1(dist(rng), dist(rng), dist(rng));
                CuckooStrongHash strong2(dist(rng), dist(rng), dist(rng));

                HashTable* ht = nullptr;
                if      (name == "Linear_Probing_Standard")
                    ht = new LinearProbing(TABLE_SIZE, seed1);
                else if (name == "Linear_Probing_PADDED")
                    ht = new LinearProbingPadded(TABLE_SIZE, seed1);
                else if (name == "Double_Hashing")
                    ht = new DoubleHashing(TABLE_SIZE, seed1, seed2);
                else if (name == "TwoWay_Chaining")
                    ht = new TwoWaySymmetric(TABLE_SIZE, seed1, seed2);
                else if (name == "Cuckoo_Symmetric")
                    ht = new CuckooSymmetric(TABLE_SIZE, strong1, strong2);
                else if (name == "Cuckoo_Asymmetric")
                    ht = new CuckooAsymmetric(TABLE_SIZE, strong1, strong2);

                // --- Insertion phase ---
                auto start_ins = std::chrono::high_resolution_clock::now();
                int  trial_probes = 0;
                bool trial_failed = false;

                for (size_t k = 0; k < num_keys; ++k) {
                    int probes = ht->insert(dataset[k]);
                    if (probes == -1) { trial_failed = true; break; }
                    trial_probes += probes;
                }
                auto end_ins = std::chrono::high_resolution_clock::now();

                if (trial_failed) {
                    // This trial used hash functions that created a cycle —
                    // record it as failed and move on to the next trial
                    raw_file << name << "," << lf << "," << t
                             << ",0,0,0,FAILED\n";
                    delete ht;
                    continue;
                }

                double ins_time = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(end_ins - start_ins).count();

                // --- Lookup phase ---
                // We look up every key we just inserted to measure the
                // average time per successful lookup at this load factor
                volatile uint32_t sink = 0;
                auto start_look = std::chrono::high_resolution_clock::now();
                for (size_t k = 0; k < num_keys; ++k) {
                    if (ht->lookup(dataset[k])) sink++;
                }
                auto end_look = std::chrono::high_resolution_clock::now();

                double look_time = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(end_look - start_look).count();

                total_insert += ins_time;
                total_lookup += look_time;
                total_probes += trial_probes;
                successful_trials++;

                raw_file << name << "," << lf << "," << t << ","
                         << (ins_time  / num_keys) << ","
                         << (look_time / num_keys) << ","
                         << ((double)trial_probes / num_keys)
                         << ",SUCCESS\n";

                delete ht;
            }

            // Write the average over all successful trials for this
            // (algorithm, load factor) pair to the summary file
            if (successful_trials == 0) {
                avg_file << name << "," << lf << ",0,0,0,FAILED\n";
                std::cout << "ALL TRIALS FAILED\n";
            } else {
                double avg_ins    = (total_insert / successful_trials) / num_keys;
                double avg_look   = (total_lookup / successful_trials) / num_keys;
                double avg_probes = (total_probes / successful_trials) / num_keys;
                avg_file << name << "," << lf << ","
                         << avg_ins   << ","
                         << avg_look  << ","
                         << avg_probes << ",SUCCESS\n";
                std::cout << "Done (" << successful_trials << "/"
                          << NUM_TRIALS << " trials OK).\n";
            }
        }
        std::cout << "------------------------------------------------------\n";
    }

    avg_file.close();
    raw_file.close();
    std::cout << "[SYSTEM] Benchmark Complete.\n";
}

int main() {
    run_benchmark();
    return 0;
}
