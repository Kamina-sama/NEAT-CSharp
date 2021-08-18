using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using System.IO.Pipes;

namespace NEAT
{
    class Program
    {
        static void Main(string[] args)
        {

            Neat neat = new Neat(3, 2, 1, 0.1, 0.3, 0.5, 0.8, 0.5, 0.7, 0.0005);
            var solution = neat.Evolve(XOR, 300, 500);

            Console.ReadKey();
            Console.WriteLine($"XOR(0,0)={Math.Round(solution.Eval(0, 0)[0])}");
            Console.WriteLine($"XOR(0,1)={Math.Round(solution.Eval(0, 1)[0])}");
            Console.WriteLine($"XOR(1,0)={Math.Round(solution.Eval(1, 0)[0])}");
            Console.WriteLine($"XOR(1,1)={Math.Round(solution.Eval(1, 1)[0])}");
        }
        public static void SquareIt(List<Genome> genomes)
        {
            Parallel.ForEach(genomes, (g) => {
                g.fitness = 100000;
                var answers = new double[10];
                answers[0] = g.Eval(0)[0];
                answers[1] = g.Eval(1)[0];
                answers[2] = g.Eval(2)[0];
                answers[3] = g.Eval(3)[0];
                answers[4] = g.Eval(4)[0];
                answers[5] = g.Eval(5)[0];
                answers[6] = g.Eval(6)[0];
                answers[7] = g.Eval(7)[0];
                answers[8] = g.Eval(8)[0];
                answers[9] = g.Eval(9)[0];

                var correct = new double[] { 0, 1, 4, 9, 16, 25, 36, 49, 64, 81};
                for (var i = 0; i < 10; ++i) g.fitness -= Math.Pow(answers[i] - correct[i], 2);
            });
        }
        public static void XOR(List<Genome> genomes)
        {
            Parallel.ForEach(genomes, (g) => {
                g.fitness = 4;
                var answers = new double[4];
                answers[0] = g.Eval(0, 0)[0];
                answers[1] = g.Eval(0, 1)[0];
                answers[2] = g.Eval(1, 0)[0];
                answers[3] = g.Eval(1, 1)[0];
                var correct = new double[4] { 0, 1, 1, 0 };
                for (var i = 0; i < 4; ++i) g.fitness -= Math.Pow(answers[i]-correct[i], 2);
            });
        }
        public static void XORSingle(Genome g)
        {
            g.fitness = 4;
            var answers = new double[4];
            answers[0] = g.Eval(0, 0)[0];
            answers[1] = g.Eval(0, 1)[0];
            answers[2] = g.Eval(1, 0)[0];
            answers[3] = g.Eval(1, 1)[0];
            var correct = new double[4] { 0, 1, 1, 0 };
            for (var i = 0; i < 4; ++i) g.fitness -= Math.Pow(answers[i] - correct[i], 2);
        }
    }
}
