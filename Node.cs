using System;
using System.Collections.Generic;
namespace NEAT
{
    class Node
    {
        bool ready = false;
        double outputNum = 0;
        public double bias = Neat.UniformRandomNumber(-30,30);
        public enum TypeOfNode {input, output, hidden}
        public readonly TypeOfNode type=TypeOfNode.hidden;
        List<Edge> input=new List<Edge>(), output=new List<Edge>();
        public Node(TypeOfNode t)
        {
            type = t;
        }
        public void Print(uint num) 
        {
            Console.WriteLine($"Node {num} with Bias={bias}");
        }
        public void Unready() { ready = false; }
        public bool PathToNonRecursive(Node dest)
        {
            Queue<Node> Q = new Queue<Node>();
            Q.Enqueue(this);
            //List<Node> Processed = new List<Node>() { origin };
            while (Q.Count > 0)
            {
                Node processing = Q.Dequeue();
                if (processing == dest) return true;
                foreach (Edge e in processing.output)
                {
                    var n = e.n2;
                    //if (!Processed.Contains(n))
                    //{
                    Q.Enqueue(n);
                    //Processed.Add(n);
                    //}
                }
            }
            return false;
        }
        public void SetInput(double d)
        {
            ready = true;
            outputNum = d;
        }
        public void AddOutput(Edge e) { output.Add(e); }
        public void AddInput(Edge e) { input.Add(e); }
        public void AssignEdgesToNode(List<Edge> edges)
        {
            input.Clear();
            output.Clear();
            foreach (Edge e in edges)
            {
                if (e.n1 == this) output.Add(e);
                if (e.n2 == this) input.Add(e);
            }
        }
        public Node Clone()
        {
            Node copy = new Node(type);
            copy.ready = ready;
            copy.outputNum = outputNum;
            copy.bias = bias;
            return copy;
        }
        public bool Disconnected()
        {
            bool disconnected_input = true, disconnected_output = true;
            foreach(var e in input)
            {
                if (e.active == true) disconnected_input = false;
            }
            foreach (var e in output)
            {
                if (e.active == true) disconnected_output = false;
            }
            return disconnected_input || disconnected_output;
        }
        public bool OutputConnectedTo(Node other)
        {
            foreach(Edge e in output)
            {
                if (e.n2 == other) return true;
            }
            return false;
        }
        public double Eval()
        {
            if (ready) return outputNum;
            double sum = bias;
            foreach(Edge e in input) if(e.active) sum += e.n1.Eval() * e.weight;
            sum = Sigmoid(sum);
            outputNum = sum;
            ready = true;
            return sum;
        }
        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }
        public static double Sigmoid(double x)
        {
            return 1 / (Math.Pow(Math.E,-x)+1);
        }
    }
}
