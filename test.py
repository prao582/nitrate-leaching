from main import *
import numpy as np



val = 1.e-8

# testing pressure odes
def test_pressure_no_mar():
    test1 = ode_model_pressure_no_mar(1,1,1,1)
    assert((test1+2) < val)

def test_pressure_mar():
    test2 = ode_model_pressure_mar(1,1,1,1,1)
    assert((test2+1.5) < val)



# testing concentration odes
def test_concentration_no_sink_no_mar():
    test3 = ode_model_concentration_no_sink_no_mar(1,1,1,1,1,1,1,1,1,1,1,1)
    assert((test3-0.5) < val)

def test_concentration_sink_no_mar():
    test4 = ode_model_concentration_sink_no_mar(1,1,1,1,1,1,1,1,1,1,1,1)
    assert((test4-0.5) < val)

def test_concentration_with_sink():
    test5 = ode_model_concentration_with_sink(1,1,1,1,1,1,1,1,1,1,1,1,)
    assert((test5-0.5) < val)

def test_concentration_with_sink_mar():
    test6 = ode_model_concentration_sink_mar(1,1,1,1,1,1,1,1,1,1,1,1)
    assert((test6-0) < val)


# testing to see if ode pressure model is working 
def test_improved_euler_pressure():
    t,p = improved_euler_pressure(f = , t0 = , t1 = , dt = , p0 = pars = )
    t_sln = 
    p_sln = 
    assert((t-t_sln) < val)
    assert((p-p_sln) < val)

# testing to see if ode concentration model is working 
def test_improved_euler_concentration():
    t,c = improved_euler_concentration(f = , t0 = , t1 = , dt = , C0 = , tdelay = , pars = ,)
    t_sln = 
    c_sln = 
    assert((t-t_sln) < val)
    assert((c-c_sln) < val)   

def main():
    test_pressure_mar()
    test_pressure_no_mar()
    test_concentration_no_sink_no_mar()
    test_concentration_sink_no_mar()
    test_concentration_with_sink()
    test_concentration_with_sink_mar()
    # test_improved_euler_pressure()
    # test_improved_euler_concentration()

if __name__=="__main__":
	main()