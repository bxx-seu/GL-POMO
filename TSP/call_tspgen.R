# Using R shell install "netgen" library:
# install.packages(BBmisc (>= 1.6),
#    mvtnorm (>= 1.0-2),
#    lhs (>= 0.10),
#    checkmate (>= 1.8.0))
# remotes::install_github("jakobbossek/netgen")
# Rscript call_tspgen.R operator point_lower point_upper ins_num seed

library("netgen")
source("./tspgen/R/utilities.R")
source("./tspgen/R/mutator.explosion.R")
source("./tspgen/R/mutator.implosion.R")
source("./tspgen/R/mutator.cluster.R")
source("./tspgen/R/mutator.compression.R")
source("./tspgen/R/mutator.expansion.R")
source("./tspgen/R/mutator.grid.R")
source("./tspgen/R/mutator.linearprojection.R")
source("./tspgen/R/mutator.rotation.R")

args <- commandArgs(TRUE)
operator <- toString(args[1])
points.lower <- as.integer(args[2])
points.upper <- as.integer(args[3])
ins.num <- as.integer(args[4])
seed <- as.integer(args[5])
choice <- toString(args[6])

options(scipen = 999)
set.seed(seed)

for (i in 1:ins.num)
{
    # points.num <- sample(points.lower:points.upper, 1, replace=TRUE)
    points.num = points.lower
    x = generateRandomNetwork(n.points = points.num, lower = 0, upper = 1)

    if (operator == "explosion")
    {
        x$coordinates = doExplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "implosion")
    {
        x$coordinates = doImplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "cluster")
    {
        x$coordinates = doClusterMutation(x$coordinates, pm=0.4)
    }
    if (operator == "compression")
    {
        x$coordinates = doCompressionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "expansion")
    {
        x$coordinates = doExpansionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "grid")
    {
        x$coordinates = doGridMutation(x$coordinates, box.min=0.3, box.max=0.3, p.rot=0, p.jitter=0, jitter.sd=0.05)
    }
    if (operator == "linearprojection")
    {
        x$coordinates = doLinearProjectionMutation(x$coordinates, pm=0.4, p.jitter=0, jitter.sd=0.05)
    }
    if (operator == "rotation")
    {
        x$coordinates = doRotationMutation(x$coordinates, pm=0.4)
    }

    x = rescaleNetwork(x, method = "global2")
    x$coordinates = x$coordinates * 1000000
    x$coordinates = round(x$coordinates, 0)
    x$coords = relocateDuplicates(x$coords)
    x$lower = 0
    x$upper = 1000000
    if (choice == "normal")
    {
        name = sprintf("./tmp/%d.tsp", i)
    }
    if (choice == "additional")
    {
        name = sprintf("./tmp/additional/%d.tsp", i)
    }
    exportToTSPlibFormat(x, name, use.extended.format=FALSE)
}
