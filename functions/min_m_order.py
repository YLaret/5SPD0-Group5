def min_m_order(am_r, m_max, significance):
    order = np.where(np.abs(am_r[m_max::]) <= significance)[0][0]
    if order == 0:
        return m_max
    else:
        return order
