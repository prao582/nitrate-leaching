from main import *
import numpy as np
from numpy.linalg import norm


testVal = 1.e-5

# Testing pressure ODEs
def test_dPdt():
    """
    Test if the function dPdt is working properly by comparing it with a known result which includes edge case. 
    """

    # Test when all inputs are 0 
    test1a = ode_model_pressure_with_sink(0, 0, 0, 0, 0, 0)
    assert((test1a - 0) < testVal)

    test2a = ode_model_pressure_with_mar(0, 0, 0, 0, 0, 0)
    assert((test2a - 0) < testVal)

    # Test when all inputs are 1
    test1b = ode_model_pressure_with_sink(1, 1, 1, 1, 1 ,1)
    assert((test1b + 2) < testVal)

    test2b = ode_model_pressure_with_mar(1, 1, 1, 1, 1 ,1)
    assert((test2b + 1.5) < testVal)

	# Test negative values
    test1c = ode_model_pressure_with_sink(-1, -1, -1, -1, -1, -1)
    assert((test1c + 2) < testVal)

    test2c = ode_model_pressure_with_mar(-1, -1, -1, -1, -1, -1)
    assert((test2c + 1.5) < testVal)

	# Test out string input 
    try:
        ode_model_pressure_with_sink('string', 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_pressure_with_mar('string', 1, 1, 1)
    except TypeError:
        pass
  
	# Test out array input 
    try:
        ode_model_pressure_with_sink(1, [1,1,1], 1, 1)
    except TypeError:
        pass

    try:
        ode_model_pressure_with_mar(1, [1,1,1], 1, 1)
    except TypeError:
        pass

    # Test out NAN variable type input 
    try:
        ode_model_pressure_with_sink(1, 1, np.NAN, 1)
    except TypeError:
        pass

    try:
        ode_model_pressure_with_mar(1, 1, np.NAN, 1)
    except TypeError:
        pass

    # Test out inf variable type input 
    try:
        ode_model_pressure_with_sink(np.inf, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_pressure_with_mar(np.inf, 1, 1, 1, 1)
    except TypeError:
        pass

    # Test missing input
    try:
        ode_model_pressure_with_sink(1)
    except TypeError:
        pass

    try:
        ode_model_pressure_with_mar(1)
    except TypeError:
        pass

# Tetsing concentration ODEs
def test_dCdt():
    """
    Test if the function dCdt is working properly by comparing it with a known result which includes edge case. 
    """

    # Test for zero division error when all inputs are 0  
    try:
        with np.errstate(divide='ignore'):
            ode_model_concentration_with_sink(0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0)
    except ZeroDivisionError:
        pass

    try:
        with np.errstate(divide='ignore'):
            ode_model_concentration_with_mar(0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0)
    except ZeroDivisionError:
        pass
    
    # Test when all inputs are 1
    test3b = ode_model_concentration_with_sink(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert((test3b - 0.5) < testVal)
    
    test4b = ode_model_concentration_with_mar(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert((test4b - 0.5) < testVal)
    
	# Test negative values
    try:
        ode_model_concentration_with_sink(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    except AssertionError:
        pass

    try:
        ode_model_concentration_with_mar(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    except AssertionError:
        pass

    # Test out string input 
    try:
        ode_model_concentration_with_sink('string', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_with_mar('string', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass
    
	# Test out array input 
    try:
        ode_model_concentration_with_sink([1,1,1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_with_mar([1,1,1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass
    
    # Test out NAN variable type input 
    try:
        ode_model_concentration_with_sink(np.NAN, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_with_mar(np.NAN, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    # Test out inf variable type input 
    try:
        ode_model_concentration_with_sink(np.inf, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_with_mar(np.inf, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    # Test missing input
    try:
        ode_model_concentration_with_sink(1)
    except TypeError:
        pass

    try:
        ode_model_concentration_with_mar(1)
    except TypeError:
        pass


# testing to see if ode pressure model is working 
def test_improved_euler_pressure():
    t,p = improved_euler_pressure(ode_model_pressure_with_mar, 10, 11, 0.5, 10, 12, 10, [-6.5,1])
    # known result for pressure 
    t_sln = [10, 10.5, 11]
    p_sln = [10, 286.25, 8193.90625]
    assert(np.linalg.norm (t - t_sln) < testVal)    
    assert(np.linalg.norm (p - p_sln) < testVal)

    # t1,p1 = improved_euler_pressure(ode_model_pressure_with_mar, 0 ,1 ,0.5 ,0 ,0, 0.5, [0,0])
    # assert(np.linalg.norm (t - t_sln) < testVal)    
    # assert(np.linalg.norm (p - p_sln) < testVal)

#     # Test backward steps   
# 	t2, p2 = solve_ode(f = , t0 = 2, t1 = 0, dt = -2, p0 = , pars = [?, ?])
# 	t2_soln = 
# 	p2_soln = 
# 	assert(np.linalg.norm(t2 - t2_soln) < testVal)
# 	assert(np.linalg.norm(p2 - p2_soln) < testVal)
    
# Testing to see if ode concentration model is working 
def test_improved_euler_concentration():
    
    # General check       
    t1,c1 = improved_euler_concentration(ode_model_concentration_with_mar, 10, 11, 0.5, 5, 1, 1, pars = [1,1,1,1,1,1,1,1] )
    t1_soln = 
    c1_soln = 
    assert(np.linalg.norm(t1 - t1_soln) < testVal)
    assert(np.linalg.norm(c1 - c1_soln) < testVal)   

#     # Test backward steps
#     t2,c2 = improved_euler_concentration(f = , t0 = 2, t1 = 0, dt = -2, C0 = , tdelay = , pars = [?, ?] )
#     t2_soln = 
#     c2_soln = 
#     assert(np.linalg.norm(t2 - t2_soln) < testVal)
#     assert(np.linalg.norm(c2 - c2_soln) < testVal)  

def main():
    #test_dPdt()
    #test_dCdt()
    #test_improved_euler_pressure()
    test_improved_euler_concentration()

if __name__=="__main__":
	main()