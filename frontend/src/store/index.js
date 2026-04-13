import React, { createContext, useContext, useReducer, useCallback } from 'react';
import { fetchOrganizedItems, fetchInsights } from '../api/wardrobeApi';

const WardrobeContext = createContext(null);

const initialState = {
    items:    [],
    insights: null,
    loading:  true,
    error:    null,
};

function reducer(state, action) {
    switch (action.type) {
        case 'LOAD_START':  return { ...state, loading: true,  error: null };
        case 'LOAD_SUCCESS': return { ...state, loading: false, items: action.items, insights: action.insights };
        case 'LOAD_ERROR':  return { ...state, loading: false, error: action.error };
        default: return state;
    }
}

export function WardrobeProvider({ children }) {
    const [state, dispatch] = useReducer(reducer, initialState);

    const loadData = useCallback(async () => {
        dispatch({ type: 'LOAD_START' });
        try {
            const [items, insights] = await Promise.all([
                fetchOrganizedItems(),
                fetchInsights(),
            ]);
            dispatch({ type: 'LOAD_SUCCESS', items, insights });
        } catch (e) {
            dispatch({ type: 'LOAD_ERROR', error: 'Could not connect to server.' });
        }
    }, []);

    return (
        <WardrobeContext.Provider value={{ ...state, loadData }}>
            {children}
        </WardrobeContext.Provider>
    );
}

export const useWardrobe = () => useContext(WardrobeContext);
