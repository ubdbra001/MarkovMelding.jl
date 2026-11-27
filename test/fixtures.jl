using JuliaBUGS
using Distributions

JuliaBUGS.@bugs_primitive Normal Logistic

const MODEL1 = compile((@bugs begin mu ~ Normal(0, 1); x ~ Normal(mu, 1) end), (x = 1.0,))
const MODEL2 = compile((@bugs begin mu ~ Logistic(5, 1); x ~ Normal(mu, 1) end), (x = 1.0,))