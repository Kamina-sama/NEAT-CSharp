using System;
using System.Collections.Generic;
using System.Linq;
namespace NEAT
{
    class Neat
    {
        public static Random rand = new Random();
        uint inputSize, outputSize;
        public List<Genome> NEATPopulation=new();
        public Genome currentBest = null;
        double delta, addNodeChance, addEdgeChance, changeWeightChance, changeWeightPower, changeBiasChance, changeBiasPower, switchChance;
        public Neat(double delta, uint inputSize, uint outputSize, double addNodeChance, double addEdgeChance, double changeWeightChance, double changeWeightPower, double changeBiasChance, double changeBiasPower, double switchChance)
        {
            this.delta = delta;
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.addNodeChance = addNodeChance;
            this.addEdgeChance = addEdgeChance;
            this.changeWeightChance = changeWeightChance;
            this.changeWeightPower = changeWeightPower;
            this.changeBiasChance = changeBiasChance;
            this.changeBiasPower = changeBiasPower;
            this.switchChance = switchChance;
        }
        public static double Normal(double mean, double stdDev)
        {
            //static Random rand = new Random(); //reuse this if you are generating many
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal =
                         mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
            return randNormal;
        }
        public static double UniformRandomNumber(double minimum, double maximum)
        {
            return rand.NextDouble() * (maximum - minimum) + minimum;
        }
        public List<Genome> InitializePopulation(uint populationSize)
        {
            NEATPopulation.Clear();
            Genome eve = new Genome(inputSize, outputSize);
            NEATPopulation.Add(eve);
            for (var i = 0; i < populationSize - 1; ++i) NEATPopulation.Add(Genome.Crossover(NEATPopulation[i], eve).Mutate(0, 0, 1, 1, 0.1, 10, 10));
            return NEATPopulation;
        }
        public void EvolveOneGeneration(uint populationSize)
        {
            var totalFitness = NEATPopulation.Sum((Genome g) => { return g.fitness; });
            var avgFitness = totalFitness / NEATPopulation.Count;
            Console.WriteLine($"Pop avg fitness= {avgFitness}");
            NEATPopulation = new List<Genome>(NEATPopulation.OrderByDescending(g => { return g.fitness; }));
            Console.WriteLine($"BEST FITNESS IS {NEATPopulation[0].fitness}");
            var best = NEATPopulation[0];
            currentBest = best.fitness > currentBest.fitness ? best : currentBest;
            var species = Speciate(NEATPopulation, 50);
            List<Genome> newPopulation = new List<Genome>();
            foreach (var specie in species) newPopulation.AddRange(Breed(specie, totalFitness, populationSize));
            NEATPopulation = newPopulation;
            Console.WriteLine($"New Population size={NEATPopulation.Count}");
        }
        public Genome Evolve(Action<List<Genome>> fitnessFunctionPopulation, uint populationSize, uint generations)
        {
            List<Genome> population = new List<Genome>();
            Genome eve = new Genome(inputSize, outputSize);
            population.Add(eve);
            for (var i = 0; i < populationSize - 1; ++i) population.Add(Genome.Crossover(population[i], eve).Mutate(0, 0, 1, 1, 0.1, 10, 10));
            Genome best = null;
            double avgFitness = 0;
            for (var i = 0; i < generations; ++i)
            {
                fitnessFunctionPopulation(population);
                var totalFitness = population.Sum((Genome g) => { return g.fitness; });
                avgFitness = totalFitness / population.Count;
                Console.WriteLine($"Pop avg fitness= {avgFitness}");
                population = new List<Genome>(population.OrderByDescending(g => { return g.fitness; }));
                Console.WriteLine($"BEST FITNESS IS {population[0].fitness}");
                best = population[0];
                var species = Speciate(population, 50);
                List<Genome> newPopulation = new List<Genome>();
                foreach (var specie in species) newPopulation.AddRange(Breed(specie, totalFitness, populationSize));
                population = newPopulation;
                Console.WriteLine($"New Population size={population.Count}");
            }
            return best;
        }
        public Genome EvolveSingle(Action<Genome> fitnessFunction, uint populationSize, uint generations)
        {
            List<Genome> population = new List<Genome>();
            Genome eve = new Genome(inputSize, outputSize);
            population.Add(eve);
            for (var i = 0; i < populationSize - 1; ++i) population.Add(Genome.Crossover(population[i], eve).Mutate(0, 0, 1, 1, 0.1, 30, 30));
            Genome best = null;
            double avgFitness = 0;
            for (var i = 0; i < generations; ++i)
            {
                foreach (var g in population) fitnessFunction(g);
                var totalFitness = population.Sum((Genome g) => { return g.fitness; });
                avgFitness = totalFitness / population.Count;
                Console.WriteLine($"Pop avg fitness= {avgFitness}");
                population = new List<Genome>(population.OrderByDescending(g => { return g.fitness; }));
                Console.WriteLine($"BEST FITNESS IS {population[0].fitness}");
                best = population[0];
                var species = Speciate(population, 50);
                List<Genome> newPopulation = new List<Genome>();
                foreach (var specie in species) newPopulation.AddRange(Breed(specie, totalFitness, populationSize));
                population = newPopulation;
                Console.WriteLine($"New Population size={population.Count}");
            }
            return best;
        }
        private List<List<Genome>> Speciate(List<Genome> population, uint maxSpeciesNum)
        {
            List<List<Genome>> species = new List<List<Genome>>();
            foreach (Genome g in population)
            {
                if (species.Count < maxSpeciesNum-1)
                {
                    bool foundSpecie = false;
                    foreach (var list in species)
                    {
                        if (g - list[0] <= delta)
                        {
                            list.Add(g);
                            foundSpecie = true;
                            break;
                        }
                    }
                    if (!foundSpecie) species.Add(new List<Genome>() { g });
                }
                else
                {
                    double minDiff = double.PositiveInfinity;
                    int index = -1;
                    foreach(var list in species)
                    {
                        if(g - list[0] < minDiff)
                        {
                            minDiff = g - list[0];
                            index = species.IndexOf(list);
                        }
                    }
                    species[index].Add(g);
                }
            }
            return species;
        }
        private List<Genome> Breed(List<Genome> specie, double totalFitness, uint populationSize)
        {
            List<Genome> newSpecieGeneration = new List<Genome>();
            double specieTotalFitness = specie.Sum((Genome g) => { return g.fitness; });
            uint numberOfIndividuals = (uint)Math.Clamp((specieTotalFitness / totalFitness) * populationSize, 0, 0.8*populationSize);
            specie=new List<Genome>(specie.OrderByDescending((Genome g) => { return g.fitness; }));
            double bestsTotalFitness = 0;
            for (int i = 0; i < (int)0.2 * specie.Count; ++i) bestsTotalFitness += specie[i].fitness;
            while (newSpecieGeneration.Count < numberOfIndividuals)
            {
                Genome g1 = null, g2 = null;
                g1 = specie[rand.Next(0, (int)0.2 * specie.Count)];
                g2 = specie[rand.Next(0, (int)0.2 * specie.Count)];
                double sum = 0;
                for(int i=0; i< (int)0.2 * specie.Count; ++i)
                {
                    sum += specie[i].fitness / bestsTotalFitness;
                    if (rand.NextDouble() <= sum) g1 = specie[i];
                    if (rand.NextDouble() <= sum) g2 = specie[i];
                }
                /*while (g1 == null)
                {
                    double sum = 0;
                    g1 = specie.Select(x =>
                    {
                        sum += x.fitness / specieTotalFitness;
                        if (rand.NextDouble() < sum)
                            return x;
                        else
                            return null;
                    }).FirstOrDefault();
                }
                while (g2 == null)
                {
                    double sum = 0;
                    g2 = specie.Select(x =>
                    {
                        sum += x.fitness / specieTotalFitness;
                        if (rand.NextDouble() < sum)
                            return x;
                        else
                            return null;
                    }).FirstOrDefault();
                }
                //Genome g1 = specie[rand.Next(Math.Min(6,specie.Count))];
                //Genome g2 = specie[rand.Next(Math.Min(6,specie.Count))];*/
                Genome child = Genome.Crossover(g1, g2).Mutate(addEdgeChance, addNodeChance, changeWeightChance, changeBiasChance, switchChance, changeWeightPower, changeBiasPower);
                newSpecieGeneration.Add(child);
            }
            return newSpecieGeneration;
        }
    }
}
