using Distributions
using NonUniformRandomVariateGeneration
using ProgressMeter

#### Gaussian Linear model (HMM)
# Yp = 1.2Xp + 1 + N(0,1)
# Xp = 0.9Xp-1 + N(0,1)

## potential
G = function(p::Int64, x::Float64)
  return exp(-(0.2*x+1.0)^2/2.0)/(sqrt(2.0*pi))
end


## particle
mutable struct Particle
  Laziness::Int64
  Eve::Int64
  Parent::Int64
  Value::Float64
end

## Algorithm
generateξ = function(N; Lazy = true, step=15)
  """
  easy SMC sampler with lazy-jump/multinomial resampling scheme.
  """
  ξ = Array{Particle,2}(undef,step,N)
  # step 0
  for i in 1:N
    ξ[1,i] = Particle(1,i,i,rand(Normal(0,1)))
  end
  # step p
  for p in 1:step-1
    if Lazy
      #println(maximum(Weight))
      ϵ = 1.0
    else
      ϵ = -1.0
    end
    @showprogress 1 "Step $p: " for i in 1:N
      if rand() < ϵ*G(p,ξ[p,i].Value)
        # non-asym case, not working for this model. 
        # ξ[p+1,i] = Particle(1,ξ[p,i].Eve,i,ξ[p,i].Value) 
        ξ[p+1,i] = Particle(1,ξ[p,i].Eve,i,rand(Normal(0.9*ξ[p,i].Value,1.0))) 
      else
        a = sampleCategorical(1, map(x -> G(p,x), getfield.(ξ[p,:],:Value)))[1]
        ξ[p+1,i] = Particle(0,ξ[p,a].Eve,a,rand(Normal(0.9*ξ[p,a].Value,1.0))) 
      end
    end
  end
  return ξ
end

γ1 = function(ξ::Array{Particle,2})
  """
  calculation of γn(1), nomalizer.
  """
  n = size(ξ)[1]
  γp = 1.0
  for p in 1:n-1
    γp *= mean(map(x -> G(p,x), getfield.(ξ[p,:],:Value)))
  end
  return γp
end

γ = function(ξ::Array{Particle,2},f::Function)
  """
  γ_n^N(⋅): unnormalized Feynman-Kac measure.

  f: test function.
  """
  n = size(ξ)[1]
  γp = 1.0
  for p in 1:n-1
    γp *= mean(map(x -> G(p,x,βtest), getfield.(ξ[p,:],:Value)))
  end
  return γp*mean(map(x -> f(x), getfield.(ξ[n,:],:Value)))
end


## variance estimators
MultinomialEstimator = function(ξ)
  """
  variance estimator of L&W.
  """
  n,N = size(ξ)
  EveSet = unique(getfield.(ξ[n,:],:Eve))
  N = Float64(N)
  m=0.0
  for eve in EveSet
      NumCommonEve = 0.0
      for i in 1:Int64(N)
          if ξ[n,i].Eve == eve
              NumCommonEve += 1.0
          end
      end
      m += NumCommonEve^2
  end
return N*γ1(ξ)^2*(1.0 - (N/(N-1))^(n-1)*(N^2 - m)/(N*(N-1)))
end 

LazyJumpEstimator = function(ξ)
  """
  new estimator with lazy-jump resampling scheme.
  stupid code by violent searching all the disjoint lines.
  """
  ϵ = 1.0 
  n,N = size(ξ)
  N = Float64(N)
  M= 0.0
  mG = Array{Float64,1}(undef,0)
  mG2 = Array{Float64,1}(undef,0)
  for p in 1:n-1
    append!(mG , mean(map(x -> G(p,x), getfield.(ξ[p,:],:Value))))
    append!(mG2 , mean(map(x -> G(p,x)^2, getfield.(ξ[p,:],:Value))))
  end
  @showprogress 1 "Computing..." for i in 1:(Int64(N)-1)
    for j in (i+1):Int64(N)
      if ξ[n,i].Eve != ξ[n,j].Eve
        M1 = 1.0
        i_prime = i 
        j_prime = j 

        for t in 1:n-1
          if ξ[n-t+1,i_prime].Laziness == 1 &&  ξ[n-t+1,j_prime].Laziness == 1 
            M1 *= mG[n-t]^2-1.0/N*mG2[n-t]
          elseif ξ[n-t+1,i_prime].Laziness == 1 &&  ξ[n-t+1,j_prime].Laziness == 0 
            M1 *= mG[n-t]^2+mG[n-t]*ϵ*ϵ*(mG2[n-t]-mG[n-t]*G(n-t,ξ[n-t,i_prime].Value))/(N-1-N*ϵ*mG[n-t]+ϵ*G(n-t,ξ[n-t,i_prime].Value))
          elseif ξ[n-t+1,i_prime].Laziness == 0 &&  ξ[n-t+1,j_prime].Laziness == 1 
            M1 *= mG[n-t]^2+mG[n-t]*ϵ*ϵ*(mG2[n-t]-mG[n-t]*G(n-t,ξ[n-t,j_prime].Value))/(N-1-N*ϵ*mG[n-t]+ϵ*G(n-t,ξ[n-t,j_prime].Value))
          elseif ξ[n-t+1,i_prime].Laziness == 0 &&  ξ[n-t+1,j_prime].Laziness == 0 
            M1 *= mG[n-t]^2 
          end
        i_prime = ξ[n-t+1,i_prime].Parent 
        j_prime = ξ[n-t+1,j_prime].Parent 
        end
        M += M1
      end
    end
  end
  M = 2.0*M
  return N*(γ1(ξ)^2- (N/(N-1))^(n-1)*M/(N*(N-1)))
end


#### consistency
N = 1024*1
ξ = generateξ(N; Lazy = true)
print("New: ")
println(LazyJumpEstimator(ξ))

N = 1024*1
ξ = generateξ(N; Lazy = false)
print("Old: ")
println(MultinomialEstimator(ξ))

#### unbiasedness
UnbiasedSimulation = function(N::Int64,NumSim::Int64,Lazy::Bool,step::Int64) 
  """
  CI is for variance estimator, i.e., it is constructed by the CLT of variance estimator, not the 
  CLT for γ_n^N.

  N: particle number
  NumSim: number of i.i.d. replicas
  Lazy: Bool
  ---------- 
  report of variance of γ_n^N(1).
  """
  mHat = Array{Float64}(undef,NumSim)
  vHat = Array{Float64}(undef,NumSim)
  @showprogress 1 "Computing..." for i in 1:NumSim
    ξ = generateξ(N; Lazy=Lazy,step = step)
    mHat[i] = γ1(ξ)
    if Lazy
      vHat[i] = LazyJumpEstimator(ξ)
    else
      vHat[i] = MultinomialEstimator(ξ)
    end
  end
  MeanEstim = mean(mHat)
  MeanV = mean(vHat)
  VarEstim =N*var(mHat)
  v_sd = sqrt(var(vHat))
  CI_length = 2*v_sd/sqrt(NumSim)
  CI_bool = abs(VarEstim - MeanV)<v_sd
  # Info:    
  sleep(0.5)
  if Lazy
    println("lazy-jump:")
  else
    println("multinomial:")
  end
  println("mean esitmation: $MeanEstim")
  println("Crude estimator: $VarEstim")
  println("mean of var estimator: $MeanV")
  #println("sd of var estimator: $v_sd")
  println("in 95% CI? $CI_bool")
end
    
UnbiasedSimulation(128,128*4,true,15)
UnbiasedSimulation(128,128*4,false,15)

#####################LOG####################
#### setting for personal pc with seconds of
#### single cpu time:
#### step = 15, N=128, Nsim=128*4 
############################################
#Computing...100%|████████████████████████████████████████████████████████████████| Time: 0:00:16
# lazy-jump:
# mean esitmation: 4.144801437553113e-8
# Crude estimator: 1.3798141021342295e-14
# mean of var estimator: 1.4945992038090894e-14
# in 95% CI? true
# 
# julia> UnbiasedSimulation(128,128*4,false)
# Computing...100%|████████████████████████████████████████████████████████████████| Time: 0:00:13
# multinomial:
# mean esitmation: 4.184739710182294e-8
# Crude estimator: 1.826341730546745e-14
# mean of var estimator: 1.7418277625338942e-14
# in 95% CI? true
