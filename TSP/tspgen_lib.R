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
