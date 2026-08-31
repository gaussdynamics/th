import numpy as np
from simulator2 import load_route_profile, make_scenario_on_route, simulate_train_tensorized

route = load_route_profile("route_generator/route_profiles/route_line0_49km.npz")
scenario = make_scenario_on_route(route, n_cars=20)
result = simulate_train_tensorized(scenario)   # result.t, result.H_hist [T,N,11], result.E_hist [T,N-1,9]

X = result.H_hist[:, :, NodeChannel.X]
V = result.H_hist[:, :, NodeChannel.V]
print("steps:", result.t.size, " H_hist:", result.H_hist.shape, " E_hist:", result.E_hist.shape)
print(f"lead distance: {X[-1,0]-X[0,0]:.1f} m")
print(f"lead speed final/max: {V[-1,0]*3.6:.1f} / {V.max()*3.6:.1f} km/h")
if result.E_hist.size:
    print(f"peak |coupler force|: {np.abs(result.E_hist[:,:,EdgeChannel.F_CPL]).max()/1e3:.1f} kN")