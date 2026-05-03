import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Units: time in days/hours , carbon in tCO2eq , health in minutes -of -life.
L = 210 # season length (days)

# Vessel parameters
t1 , t2 = 2, 3 # trips per day per vessel (type I, type II)
cap1 , cap2 = 300, 27 # passengers per trip (type I, type II)
th1 , th2 = 1.5, 2 # hours spent with whales per trip (type I, type II)

ov1 , ov2 = 0.61, 0.8 # proportion of time overlapping with foraging (type I, type II)

p_enc = 0.05 # probability of repeatedly encountering the same whales

e1 , e2 = 2.36*10**(-3), 41.63*10**(-3) # GHG per passenger-trip (tCO2eq per passenger trip , type I, type II)

hb1 , hb2 = 43.95/(th1), 118.6/(th2) # minutes -of -life gained per passenger-hour with whales (type I, type II)

# Biodiversity logistic regression coefficients
a = -4.97 # intercept of logistic model
b = 1.76 # coefficient on foraging overlap
c = 0.0036 # coefficient on whale time (hours)

# Environmental boundary (max foetal growth loss)
B_max = 0.184 # 18.4%
P_start = 300000 # initial passenger demand

# Weights
w_H = 2.0 # weight on health benefit term
w_P = 1.0 # weight on passengers served term
w_C = 1.0 # weight on carbon penalty term
w_B = 2.0 # weight on biodiversity penalty term
w_cap = 20.0 # weight on excess capacity penalty term

def passengers(n1 , n2, L=130, t1=2, t2=2, cap1=20, cap2=100):
    Ptot = L * (n1 * t1 * cap1 + n2 * t2 * cap2)
    P1 = L * n1 * t1 * cap1
    P2 = L * n2 * t2 * cap2
    return Ptot, P1, P2

def carbon(n1 , n2, L=130, t1=2, t2=2, cap1=20, cap2=100):
    P1 = L * n1 * t1 * cap1
    P2 = L * n2 * t2 * cap2
    return P1 * e1 + P2 * e2, P1*e1, P2*e2

def health(n1, n2, hb1=1, hb2=1, th1=0.5, th2=0.2, L=130, t1=2, t2=2, cap1=20, cap2=100):
    h1 = L * n1 * t1 * cap1 * th1
    h2 = L * n2 * t2 * cap2 * th2
    return hb1 * h1 + hb2 * h2, hb1 * h1, hb2 * h2

def biodiversity(n1 , n2, L=130, t1=2, t2=2, th1=0.5, th2=0.2, ov1=0.5, ov2=0.3, p_enc=0.5, a=-4.97, b=1.76, c=0.0036, cap1=20, cap2=100):
    #Foetal growth loss fraction from whale exposure and overlap
    # total time per season with whales
    T_whale = L * (n1 * t1 * th1 + n2 * t2 * th2)

    # passenger -weighted overlap
    P1 = n1 * t1 * cap1
    P2 = n2 * t2 * cap2
    if (P1 + P2) == 0:
        overlap_eff = 0.0
    else:
        overlap_eff = (ov1 * P1 + ov2 * P2) / (P1 + P2)

    # logistic regression
    x = a + b * overlap_eff + c * T_whale * p_enc
    B = 1.0 / (1.0 + np.exp(-x))
    return B

def combined_objective(n1 , n2 , w_H=1 , w_P=1 , w_C=1 , w_B=1, w_cap=1, P_demand=1, H_ref=1 , P_ref=1 , C_ref=1 , B_ref=1, L=130, t1=2, t2=2, cap1=20, cap2=100, th1=0.5, th2=0.2, hb1=1, hb2=1, ov1=0.5, ov2=0.3, p_enc=0.5, a=-4.97, b=1.76, c=0.0036):

    P = passengers(n1 , n2, L, t1, t2, cap1, cap2)[0] # total passengers
    C = carbon(n1 , n2, L, t1, t2, cap1, cap2)[0] # total carbon emissions
    H = health(n1 , n2, hb1=hb1, hb2=hb2, th1=th1, th2=th2, L=L, t1=t1, t2=t2, cap1=cap1, cap2=cap2)[0] # total health benefit
    B = biodiversity(n1 , n2, L, t1, t2, th1, th2, ov1, ov2, p_enc, a, b, c, cap1, cap2) # biodiversity impact (foetal growth loss fraction)

    Hs = H / H_ref if H_ref > 0 else 0.0
    Ps = P / P_ref if P_ref > 0 else 0.0
    Cs = C / C_ref if C_ref > 0 else 0.0
    Bs = B / B_ref if B_ref > 0 else 0.0
    capacity_excess = max(P - P_demand, 0.0)
    capacity_penalty = capacity_excess / P_demand if P_demand > 0 else 0.0


    print(f"Debug: P={Ps:.0f}, C={Cs:.2f}, H={Hs:.2f}, B={Bs:.4f}, Capacity Penalty={capacity_penalty:.4f}")
    F = w_H * Hs + w_P * Ps - w_C * Cs - w_B * Bs - w_cap * capacity_penalty
    return F, (P, C, H, B)

def passenger_demand(year , P0=10 , growth_rate =0.05):
    return P0 * (1.0 + growth_rate)**( year - 1)

# Evaluate combined objective over years
years = np.arange(1, 36)  # years 1 to 25
growth_rate = 0.05  # 5% annual growth in passenger demand

# Compute reference metrics from initial reference fleet
n1_ref, n2_ref = 3, 2
P_ref = passengers(n1_ref, n2_ref, L, t1, t2, cap1, cap2)[0]
C_ref = carbon(n1_ref, n2_ref, L, t1, t2, cap1, cap2)[0]
H_ref = health(n1_ref, n2_ref, hb1, hb2, th1, th2, L, t1, t2, cap1, cap2)[0]
B_ref = biodiversity(n1_ref, n2_ref, L, t1, t2, th1=th1, th2=th2, ov1=ov1, ov2=ov2, p_enc=p_enc, a=a, b=b, c=c, cap1=cap1, cap2=cap2)

print("Reference fleet (n1_ref, n2_ref) =", n1_ref, n2_ref)
print("P_ref =", P_ref, "passengers per season")
print("C_ref =", C_ref)
print("H_ref =", H_ref)
print("B_ref =", B_ref)

# Optimization setup
demand_values = []
objective_values = []
n1_optimal = []
n2_optimal = []
b_values = []
p_capacity = []

def neg_objective(x):
    """Negative combined objective (for minimization)"""
    n1, n2 = x
    if n1 < 0 or n2 < 0:
        return 1e10
    F, _ = combined_objective(n1, n2, 
                              w_H=w_H, w_P=w_P, w_C=w_C, w_B=w_B,
                              w_cap=w_cap, P_demand=current_demand,
                              H_ref=H_ref, P_ref=P_ref, C_ref=C_ref, B_ref=B_ref,
                              L=L, t1=t1, t2=t2, cap1=cap1, cap2=cap2,
                              th1=th1, th2=th2, hb1=hb1, hb2=hb2,
                              ov1=ov1, ov2=ov2, p_enc=p_enc,
                              a=a, b=b, c=c)
    return -F

def constraint_biodiversity(x):
    """Constraint: biodiversity impact <= B_max"""
    n1, n2 = x
    if n1 < 0 or n2 < 0:
        return -1e10
    B = biodiversity(n1, n2, L, t1, t2, th1, th2, ov1, ov2, p_enc, a, b, c, cap1, cap2)
    return B_ref - B  # should be >= 0

def constraint_demand(x, year_demand):
    """Constraint: fleet must serve at least the yearly demand"""
    n1, n2 = x
    if n1 < 0 or n2 < 0:
        return -1e10
    P_total, _, _ = passengers(n1, n2, L, t1, t2, cap1, cap2)
    return P_total - year_demand  # should be >= 0

current_demand = P_start

print("\nYear-by-year optimization (integer fleet sizes):")
print("Year\tDemand\t\tN1\tN2\tCapacity\tObjective\tBiodiversity")

for year in years:
    demand = passenger_demand(year, P0=P_start, growth_rate=growth_rate)
    demand_values.append(demand)
    current_demand = demand
    
    # Initial guess based on reference fleet scaled by demand
    scale_factor = demand / P_ref if P_ref > 0 else 1.0
    x0 = np.array([n1_ref * scale_factor, n2_ref * scale_factor])
    
    # Constraints: biodiversity AND demand satisfaction
    constraints = [
        {'type': 'ineq', 'fun': constraint_biodiversity},
        {'type': 'ineq', 'fun': constraint_demand, 'args': (demand,)}
    ]
    
    # Bounds: at least 0 vessels, at most reasonable max (e.g., 50 of each)
    bounds = [(0, 50), (0, 50)]
    
    # Optimize with continuous variables first
    result = minimize(neg_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})
    
    # Round to nearest integers
    n1_continuous, n2_continuous = result.x if result.success else x0
    n1_opt = int(np.round(n1_continuous))
    n2_opt = int(np.round(n2_continuous))
    
    # Check if rounded solution is feasible; if not, try nearby integer combinations
    feasible = False
    best_obj = -np.inf
    
    for dn1 in range(-1, 2):
        for dn2 in range(-1, 2):
            n1_test = n1_opt + dn1
            n2_test = n2_opt + dn2
            
            if n1_test < 0 or n2_test < 0:
                continue
            
            P_total, _, _ = passengers(n1_test, n2_test, L, t1, t2, cap1, cap2)
            B_test = biodiversity(n1_test, n2_test, L, t1, t2, th1, th2, ov1, ov2, p_enc, a, b, c, cap1, cap2)
            
            # Check constraints
            if P_total >= demand and B_test <= B_max:
                F_test, _ = combined_objective(n1_test, n2_test,
                                              w_H=w_H, w_P=w_P, w_C=w_C, w_B=w_B,
                                              w_cap=w_cap, P_demand=demand,
                                              H_ref=H_ref, P_ref=P_ref, C_ref=C_ref, B_ref=B_ref,
                                              L=L, t1=t1, t2=t2, cap1=cap1, cap2=cap2,
                                              th1=th1, th2=th2, hb1=hb1, hb2=hb2,
                                              ov1=ov1, ov2=ov2, p_enc=p_enc,
                                              a=a, b=b, c=c)
                
                if F_test > best_obj:
                    best_obj = F_test
                    n1_opt, n2_opt = n1_test, n2_test
                    feasible = True
    
    if not feasible:
        # If no feasible nearby solution, scale up from continuous solution
        n1_opt = int(np.ceil(n1_continuous))
        n2_opt = int(np.ceil(n2_continuous))
    
    P_capacity, _, _ = passengers(n1_opt, n2_opt, L, t1, t2, cap1, cap2)
    F_opt, (P_opt, C_opt, H_opt, B_opt) = combined_objective(n1_opt, n2_opt,
                                                              w_H=w_H, w_P=w_P, w_C=w_C, w_B=w_B,
                                                              w_cap=w_cap, P_demand=demand,
                                                              H_ref=H_ref, P_ref=P_ref, C_ref=C_ref, B_ref=B_ref,
                                                              L=L, t1=t1, t2=t2, cap1=cap1, cap2=cap2,
                                                              th1=th1, th2=th2, hb1=hb1, hb2=hb2,
                                                              ov1=ov1, ov2=ov2, p_enc=p_enc,
                                                              a=a, b=b, c=c)
    
    n1_optimal.append(n1_opt)
    n2_optimal.append(n2_opt)
    objective_values.append(F_opt)
    b_values.append(B_opt)
    p_capacity.append(P_capacity)
    
    print(f"{year}\t{demand:.0f}\t\t{n1_opt}\t{n2_opt}\t{P_capacity:.0f}\t{F_opt:.4f}\t\t{B_opt:.4f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(years, demand_values, 'b-o', label='Passenger Demand')
axes[0, 0].plot(years, p_capacity, 'g--s', label='Fleet Capacity')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Passengers')
axes[0, 0].set_title('Passenger Demand vs Fleet Capacity Over Time')
axes[0, 0].grid(True)
axes[0, 0].legend()

axes[0, 1].plot(years, n1_optimal, 'g-o', label='Type I Vessels (n1)', linewidth=2, markersize=8)
axes[0, 1].plot(years, n2_optimal, 'm-s', label='Type II Vessels (n2)', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Number of Vessels')
axes[0, 1].set_title('Optimal Fleet Composition Over Time (Integer)')
axes[0, 1].grid(True)
axes[0, 1].legend()

axes[1, 0].plot(years, objective_values, 'r-o', label='Combined Objective', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Objective Value')
axes[1, 0].set_title('Combined Objective Over Time (Optimized)')
axes[1, 0].grid(True)
axes[1, 0].legend()

axes[1, 1].plot(years, b_values, 'purple', marker='o', label='Biodiversity Impact', linewidth=2, markersize=8)
axes[1, 1].axhline(y=B_max, color='r', linestyle='--', label=f'B_max = {B_max}', linewidth=2)
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Foetal Growth Loss Fraction')
axes[1, 1].set_title('Biodiversity Impact Over Time')
axes[1, 1].grid(True)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('objective_optimization.png', dpi=150)
plt.show()






