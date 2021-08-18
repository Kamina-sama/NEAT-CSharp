namespace NEAT
{
    class Edge
    {
        public Edge(uint innov, Node n1, Node n2, double weight, bool active=true)
        {
            innovation_num = innov;
            this.active = active;
            this.n1 = n1;
            this.n2 = n2;
            this.weight = weight;
        }
        public uint innovation_num;
        public bool active=true;
        public Node n1,n2;
        public double weight=1;
    }
}
