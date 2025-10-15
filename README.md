## Modelling GNUTELLA-like P2P File Transfer Networks
This project provides a modelling framework for GNUTELLA-like peer-to-peer (P2P) file transfer and distribution networks. It models network behavior, agent interactions using Gossip messaging to study the features of P2P as a complex system.

### Features
- Graph Generation - creates initial overlay Network.
- Simulates P2P network topologies and client roles
    Seeder | Leecher | Hybrid
- Models node failures and recovery
- Node/Agent Routing of Query/QueryHit
- Analyses data distribution efficiency and emergent features

### Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/jbwod/COMP4403-Project.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the framework with the included Demo Jupyter Notebook.
4. Generate a Graph with desired Parameters and topology.
<img width="593" height="219" alt="image" src="https://github.com/user-attachments/assets/d9d806ea-4e29-430b-9de5-cfd772a9745a" />
 
5. Apply a File and Node Distribution
<img width="877" height="473" alt="Screenshot 2025-10-14 at 7 13 20 pm" src="https://github.com/user-attachments/assets/25bfd63b-1da9-4717-9171-6a48fe5eb57e" />

7.  Configure your Simulation Parameters
   - Search Mode | Realistic (All Agents, one Round) or Single (one Agent, one Round)
   - Neighbor Selection | Random (K random) or `Bandwidth (K filtered by Edge Weight)
   - TTL - Query Time To Live
   - _K_ infection - Query Forwarding
   - Single Agent (Only this Node make queries - Optional)
8. Run the Model and watch it Transfer
![rounds](https://github.com/user-attachments/assets/0c258961-fb6d-4d49-9192-65f914f7e502)
