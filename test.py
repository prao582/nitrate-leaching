from main import *
import numpy as np



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

# testing the nuerical pressure model
def test_pressure_analytical():
    t1,pn1 = pressure_analytical(0,1,0.5,1)
    # known result for pressure
    pn_sln1 = [1.244603014e-55, 4.578638612e-56]
    assert((pn1[0]-pn_sln1[0]) < testVal)
    assert((pn1[1]-pn_sln1[1]) < testVal)
    
    t2,pn2 = pressure_analytical(10,11,0.5,-6.5)
    # known result for pressure
    pn_sln2 = [35.75308289, 23780.86394]
    assert((pn2[0]-pn_sln2[0]) < testVal)
    assert((pn2[1]-pn_sln2[1]) < testVal)



# testing to see if ode pressure model is working 
def test_improved_euler_pressure():
    t,p = improved_euler_pressure(ode_model_pressure_with_mar, 10, 11, 0.5, 10, [-6.5,1])
    # known result for pressure 
    p_sln = [10, 997.6789148, 657944.0453]
    assert((p[0]-p_sln[0]) < testVal)
    assert((p[1]-p_sln[1]) < testVal)
    assert((p[2]-p_sln[2]) < testVal)
    

#     # Test backward steps   
# 	t2, p2 = solve_ode(f = , t0 = 2, t1 = 0, dt = -2, p0 = , pars = [?, ?])
# 	t2_soln = 
# 	p2_soln = 
# 	assert(np.linalg.norm(t2 - t2_soln) < testVal)
# 	assert(np.linalg.norm(p2 - p2_soln) < testVal)
    
# # Testing to see if ode concentration model is working 
# def test_improved_euler_concentration():
    
#     # General check
#     t1,c1 = improved_euler_concentration(f = , t0 = , t1 = , dt = , C0 = , tdelay = , pars = [?, ?] )
#     t1_soln = 
#     c1_soln = 
#     assert(np.linalg.norm(t1 - t1_soln) < testVal)
#     assert(np.linalg.norm(c1 - c1_soln) < testVal)   

#     # Test backward steps
#     t2,c2 = improved_euler_concentration(f = , t0 = 2, t1 = 0, dt = -2, C0 = , tdelay = , pars = [?, ?] )
#     t2_soln = 
#     c2_soln = 
#     assert(np.linalg.norm(t2 - t2_soln) < testVal)
#     assert(np.linalg.norm(c2 - c2_soln) < testVal)  

def main():
    #test_dPdt()
    #test_dCdt()
    #test_pressure_analytical()
    test_improved_euler_pressure()
    # test_improved_euler_concentration()

if __name__=="__main__":
	main()