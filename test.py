from main import *
import numpy as np



testVal = 1.e-10

# Testing pressure ODEs
def test_dPdt():
    """
    Test if the function dPdt is working properly by comparing it with a known result which includes edge case. 
    """

    # Test when all inputs are 0 
    test1a = ode_model_pressure_no_mar(0, 0, 0, 0)
    assert((test1a - 0) < testVal)

    test2a = ode_model_pressure_mar(0, 0, 0, 0, 0)
    assert((test2a - 0) < testVal)

    # Test when all inputs are 1
    test1b = ode_model_pressure_no_mar(1, 1, 1, 1)
    assert((test1b + 2) < testVal)

    test2b = ode_model_pressure_mar(1, 1, 1, 1, 1)
    assert((test2b + 1.5) < testVal)

	# Test negative values
    test1c = ode_model_pressure_no_mar(-1, -1, -1, -1)
    assert((test1c + 2) < testVal)

    test2c = ode_model_pressure_mar(-1, -1, -1, -1, -1)
    assert((test2c + 1.5) < testVal)

	# Test out string input 
    try:
        test1d = ode_model_pressure_no_mar('string', 1, 1, 1)
    except TypeError:
        pass

    try:
        test2d = ode_model_pressure_mar('string', 1, 1, 1)
    except TypeError:
        pass
  
	# Test out array input 
    try:
        test1e = ode_model_pressure_no_mar(1, [1,1,1], 1, 1)
    except TypeError:
        pass

    try:
        test2e = ode_model_pressure_mar(1, [1,1,1], 1, 1)
    except TypeError:
        pass

    # Test out NAN variable type input 
    try:
        test1f = ode_model_pressure_no_mar(1, 1, np.NAN, 1)
    except TypeError:
        pass

    try:
        test2f = ode_model_pressure_mar(1, 1, np.NAN, 1)
    except TypeError:
        pass

    # Test out inf variable type input 
    try:
        test1g = ode_model_pressure_no_mar(np.inf, 1, 1, 1)
    except TypeError:
        pass

    try:
        test2g = ode_model_pressure_mar(np.inf, 1, 1, 1, 1)
    except TypeError:
        pass

    # Test missing input
    try:
        test1h = ode_model_pressure_no_mar(1)
    except TypeError:
        pass

    try:
        test2h = ode_model_pressure_mar(1)
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
            ode_model_concentration_no_sink_no_mar(0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0)
    except ZeroDivisionError:
        pass

    try:
        with np.errstate(divide='ignore'):
            ode_model_concentration_sink_no_mar(0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0)
    except ZeroDivisionError:
        pass

    try:
        with np.errstate(divide='ignore'):
            ode_model_concentration_sink_mar(0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0)
    except ZeroDivisionError:
        pass
    
    # Test when all inputs are 1
    test3b = ode_model_concentration_no_sink_no_mar(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert((test3b - 0.5) < testVal)
    
    test4b = ode_model_concentration_sink_no_mar(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert((test4b - 0.5) < testVal)

    test5b = ode_model_concentration_no_sink_no_mar(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert((test5b - 0.5) < testVal)

    test6b = ode_model_concentration_sink_mar(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert((test6b - 0) < testVal)    
    
	# Test negative values
    try:
        ode_model_concentration_no_sink_no_mar(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    except AssertionError:
        pass

    try:
        ode_model_concentration_sink_no_mar(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    except AssertionError:
        pass

    try:
        ode_model_concentration_no_sink_no_mar(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    except AssertionError:
        pass

    try:
        ode_model_concentration_sink_mar(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    except AssertionError:
        pass
 
    # Test out string input 
    try:
        ode_model_concentration_no_sink_no_mar('string', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_no_mar('string', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass
    
    try:
        ode_model_concentration_no_sink_no_mar('string', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_mar('string', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

	# Test out array input 
    try:
        ode_model_concentration_no_sink_no_mar([1,1,1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_no_mar([1,1,1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass
    
    try:
        ode_model_concentration_no_sink_no_mar([1,1,1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_mar([1,1,1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass


    # Test out NAN variable type input 
    try:
        ode_model_concentration_no_sink_no_mar(np.NAN, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_no_mar(np.NAN, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass
    
    try:
        ode_model_concentration_no_sink_no_mar(np.NAN, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_mar(np.NAN, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass
    
    # Test out inf variable type input 
    try:
        ode_model_concentration_no_sink_no_mar(np.inf, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_no_mar(np.inf, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass
    
    try:
        ode_model_concentration_no_sink_no_mar(np.inf, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_mar(np.inf, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    except TypeError:
        pass

    # Test missing input
    try:
        ode_model_concentration_no_sink_no_mar(1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_no_mar(1)
    except TypeError:
        pass
    
    try:
        ode_model_concentration_no_sink_no_mar(1)
    except TypeError:
        pass

    try:
        ode_model_concentration_sink_mar(1)
    except TypeError:
        pass


#testing to see if ode pressure model is working 
def test_improved_euler_pressure():
    t,p = improved_euler_pressure(f = , t0 = , t1 = , dt = , p0 = pars = )
    t_sln = 
    p_sln = 
    assert((t-t_sln) < testVal)
    assert((p-p_sln) < testVal)

#testing to see if ode concentration model is working 
def test_improved_euler_concentration():
    t,c = improved_euler_concentration(f = , t0 = , t1 = , dt = , C0 = , tdelay = , pars = ,)
    t_sln = 
    c_sln = 
    assert((t-t_sln) < testVal)
    assert((c-c_sln) < testVal)   

def main():
    test_dPdt()
    test_dCdt()
    # test_improved_euler_pressure()
    # test_improved_euler_concentration()

if __name__=="__main__":
	main()