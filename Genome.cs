using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NEAT
{
    class Genome
    {
        public static double c1 = 1, c2 = 0.5, c3 = 0.5;
        public double fitness = 0;
        static Random rad = new Random();
        static uint innov = 0;
        int biggestInnovIHave = -1;
        List<Node> nodes;
        List<Node> inputNodes;
        List<Node> outputNodes;
        List<Edge> edges;
        public Genome Clone()
        {
            Genome copy = new Genome((uint)0, (uint)0);
            for (int i = 0; i < inputNodes.Count; ++i) copy.inputNodes.Add(inputNodes[i].Clone());
            for (int i = 0; i < outputNodes.Count; ++i) copy.outputNodes.Add(outputNodes[i].Clone());
            copy.nodes.AddRange(copy.inputNodes);
            copy.nodes.AddRange(copy.outputNodes);
            for (int i = inputNodes.Count + outputNodes.Count; i < nodes.Count; ++i) copy.nodes.Add(nodes[i].Clone());
            foreach (Edge e in edges)
            {
                copy.edges.Add(new Edge(e.innovation_num, copy.nodes[nodes.IndexOf(e.n1)], copy.nodes[nodes.IndexOf(e.n2)], e.weight, e.active));
            }
            copy.biggestInnovIHave = biggestInnovIHave;
            copy.fitness = fitness;
            copy.AssignEdgesToNodes();
            return copy;
        }
        private void AssignEdgesToNodes()
        {
            foreach (var node in nodes) node.AssignEdgesToNode(edges);
        }
        public static double operator -(Genome g1, Genome g2)
        {
            Dictionary<uint, Tuple<Edge, Edge>> pairsOfEdges = new Dictionary<uint, Tuple<Edge, Edge>>();
            foreach (Edge e in g1.edges)
            {
                pairsOfEdges.Add(e.innovation_num, new Tuple<Edge, Edge>(e, null));
            }
            foreach (Edge e in g2.edges)
            {
                if (pairsOfEdges.ContainsKey(e.innovation_num))
                {
                    pairsOfEdges[e.innovation_num] = new Tuple<Edge, Edge>(pairsOfEdges[e.innovation_num].Item1, e);
                }
                else pairsOfEdges.Add(e.innovation_num, new Tuple<Edge, Edge>(null, e));
            }
            uint D = 0;
            uint E = 0;
            uint total = 0;
            List<double> differences = new List<double>();
            foreach (var (k, v) in pairsOfEdges)
            {
                if (v.Item2 == null)
                {
                    if (g1.biggestInnovIHave < k) ++E;
                    else ++D;
                }
                else if (v.Item1 == null)
                {
                    if (g2.biggestInnovIHave < k) ++E;
                    else ++D;
                }
                else
                {
                    differences.Add(Math.Abs(v.Item1.weight - v.Item2.weight));
                    ++total;
                }
            }

            //Get the delta=c1*D+c2*E+c3*^W
            //D=number of disjoint Genes
            //E=Number of Excess Genes
            //^W=Average Difference in Weights= sumof(wi1-wi2)/quantity of compatible weights
            //c1, c2 and c3 are coefficients that give more or less weights to these factors
            //My implementation: fuck this, we doing this my way: c1(D)+c2^W where D is disjoint and excess
            if (total == 0)
            {
                total = 1;
            }
            return c1 * D + c2 * E + c3 * differences.Sum() / total;
        }
        public void Print()
        {
            Console.WriteLine($"Inputs: {inputNodes.Count}");
            Console.WriteLine($"Outputs: {outputNodes.Count}");
            Console.WriteLine($"Total nodes: {nodes.Count}");
            for (var i = 0; i < nodes.Count; ++i)
            {
                nodes[i].Print((uint)i);
            }
            Console.WriteLine($"Edges (from, to): ");
            foreach (var e in edges)
            {
                Console.Write($"({nodes.IndexOf(e.n1)}, {nodes.IndexOf(e.n2)}) Enabled:{e.active} Weight:{e.weight} ");
                if (e.n1.type == Node.TypeOfNode.input) Console.Write($"Input Node: {inputNodes.IndexOf(e.n1)} ");
                if (e.n2.type == Node.TypeOfNode.output) Console.Write($"Output Node: {outputNodes.IndexOf(e.n2)} ");
                Console.WriteLine();
            }
        }
        public Genome(uint inputSize, uint outputSize)
        {
            inputNodes = new List<Node>();
            for (var i = 0; i < inputSize; ++i) inputNodes.Add(new Node(Node.TypeOfNode.input));
            outputNodes = new List<Node>();
            for (var i = 0; i < outputSize; ++i) outputNodes.Add(new Node(Node.TypeOfNode.output));
            nodes = new List<Node>(inputNodes);
            nodes.AddRange(outputNodes);
            edges = new List<Edge>();
            foreach (var i in inputNodes)
            {
                foreach (var o in outputNodes)
                {
                    Edge e = new Edge(innov++, i, o, Neat.UniformRandomNumber(-30, 30), true);
                    i.AddOutput(e);
                    o.AddInput(e);
                    edges.Add(e);
                }
            }
        }
        public Genome Mutate(double addEdgeChance, double addNodeChance, double changeWeightChance, double changeBiasChance, double switchChance, double changeWeightPower, double changeBiasPower)
        {
            if (rad.NextDouble() < addEdgeChance) AddEdge();
            if (rad.NextDouble() < addNodeChance) AddNode();
            foreach (Node n in nodes.ToArray())
            {
                if (n.type != Node.TypeOfNode.input && rad.NextDouble() < changeBiasChance)
                {
                    n.bias += Neat.Normal(0, changeBiasPower);
                }
                //BIAS REPLACE RATE
                if (n.type != Node.TypeOfNode.input && rad.NextDouble() < 0.01)
                {
                    n.bias = Neat.UniformRandomNumber(-30, 30);
                }
            }
            foreach (Edge edge in edges.ToArray())
            {
                if (rad.NextDouble() < changeWeightChance)
                {
                    edge.weight += Neat.Normal(0, changeWeightPower);
                }
                if (rad.NextDouble() < switchChance)
                {
                    //IT SHOULD ONLY DEACTIVATE IF DOING SO DOENST DISCONNECT THE NEURON (BOTH I/O)!!!!!
                    edge.active = !edge.active;
                    //if this disconnected a hidden node, revert change
                    if ((edge.n1.type == Node.TypeOfNode.hidden && edge.n1.Disconnected()) || (edge.n2.type == Node.TypeOfNode.hidden && edge.n2.Disconnected())) edge.active = !edge.active;
                }
                //WEIGHT REPLACE RATE
                if (rad.NextDouble() < 0.01)
                {
                    edge.weight = Neat.UniformRandomNumber(-30, 30);
                }
            }
            fitness = 0;
            return this;
        }
        public static Genome Crossover(Genome first, Genome second)
        {
            if (second.fitness > first.fitness)
            {
                Genome temp = second;
                second = first;
                first = temp;
            }
            Genome child = first.Clone();
            Dictionary<uint, Edge> childEdges = new Dictionary<uint, Edge>();
            for (int i = 0; i < first.edges.Count(); ++i) childEdges.Add(child.edges[i].innovation_num, child.edges[i]); //At first, child edges are exactly like the best parent
            for (int i = 0; i < second.edges.Count(); ++i)
            {
                //THEN, for each innov num the best and second best parent have in common, there is a chance that the second best's gene will take it's place:
                if (childEdges.ContainsKey(second.edges[i].innovation_num) && rad.NextDouble() < second.fitness / (first.fitness + second.fitness))
                {
                    var index1 = second.nodes.IndexOf(second.edges[i].n1);
                    var index2 = second.nodes.IndexOf(second.edges[i].n2);
                    child.edges[i].n1 = child.nodes[index1];
                    child.edges[i].n2 = child.nodes[index2];
                    child.edges[i].weight = second.edges[i].weight;
                    child.edges[i].active = second.edges[i].active;
                }
            }
            child.AssignEdgesToNodes();
            return child;
        }
        public void AddNode()
        {
            if (edges.Count > 0)
            {
                Edge edge_to_be_split = edges[rad.Next(edges.Count)];
                Node new_node = new Node(Node.TypeOfNode.hidden);
                nodes.Add(new_node);
                var n1 = edge_to_be_split.n1;
                var n2 = edge_to_be_split.n2;
                edge_to_be_split.active = false;

                Edge new_edge1 = new Edge(innov++, n1, new_node, edge_to_be_split.weight);
                edges.Add(new_edge1);
                Edge new_edge2 = new Edge(innov++, new_node, n2, 1);
                edges.Add(new_edge2);

                new_node.AddInput(new_edge1);
                new_node.AddOutput(new_edge2);
                biggestInnovIHave = (int)innov - 1;
            }
        }
        public void AddEdge()
        {
            Node n1 = nodes[rad.Next(nodes.Count)], n2 = nodes[nodes.Count - 1];
            //THIS LOOP IS PROBELMATIC
            while (n1 == n2 || n1.type == Node.TypeOfNode.output) n1 = nodes[rad.Next(nodes.Count)];
            uint maxTries = 5;
            uint tries = 0;
            while ((n1 == n2 || n1.OutputConnectedTo(n2) || n2.PathToNonRecursive(n1)) && tries < maxTries)
            {
                n2 = nodes[rad.Next(inputNodes.Count, nodes.Count)];
                ++tries;
            }
            if (n1 == n2 || n1.OutputConnectedTo(n2) || n2.PathToNonRecursive(n1)) return;
            Edge e = new Edge(innov++, n1, n2, Neat.Normal(0, 2));
            edges.Add(e);
            n1.AddOutput(e);
            n2.AddInput(e);
            biggestInnovIHave = (int)innov - 1;
        }
        public double[] Eval(params double[] args)
        {
            //The way im doing here makes it so that the input nodes have linear activation F(x)=x. Dunno if that's a good thing but oh well.
            foreach (Node n in nodes) n.Unready();
            for (var i = 0; i < inputNodes.Count; ++i) inputNodes[i].SetInput(args[i]);
            double[] results = new double[outputNodes.Count];
            for (var i = 0; i < outputNodes.Count; ++i) results[i] = outputNodes[i].Eval();
            return results;
        }
    }
}
