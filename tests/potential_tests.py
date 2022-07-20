def test_exploit_policy():
    dummy_state = [0, 0, 1]
    dummy_Q = np.random.randint(0, 100, size=(K+1, 2, 2, 3))
    dummy_Q[0,0,1,:]
    exploit_policy(dummy_state, dummy_Q)


