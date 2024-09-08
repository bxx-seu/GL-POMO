# args <- commandArgs(TRUE)
# points.lower <- as.integer(args[1])
# points.upper <- as.integer(args[2])
# clu.lower <- as.integer(args[3])
# clu.upper <- as.integer(args[4])
# ins.num <- as.integer(args[5])
# seed <- as.integer(args[6])

set.seed(seed)

dataset <- list()
index = 1
for (clu_num in clu.lower:clu.upper)
{
    for (i in 1:ins.num)
    {
        # points.num <- sample(points.lower:points.upper, 1, replace=TRUE)
        x = generateClusteredNetwork(n.points = points.num, n.cluster = clu_num, out.of.bounds.handling = "mirror", upper=100)
        x = rescaleNetwork(x, method = "global2")
        x$coordinates = x$coordinates * 1000000
        x$coordinates = round(x$coordinates, 0)
        x$lower = 0
        x$upper = 1000000
        # name = sprintf("../data/TSP/cl/%d.tsp", index)
        # exportToTSPlibFormat(x, name, use.extended.format=FALSE)
        dataset[i] <- x[1]
        index = index + 1
    }
}