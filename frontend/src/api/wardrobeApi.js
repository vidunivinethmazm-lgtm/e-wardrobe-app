import axios from 'axios';
import BASE_URL from './config';

const api = axios.create({ baseURL: BASE_URL, timeout: 10000 });

export const fetchOrganizedItems = async () => {
    const res = await api.get('/items/organized');
    return res.data;
};

export const fetchInsights = async () => {
    const res = await api.get('/items/insights');
    return res.data; // { dirty_count, underused: [ids], overused: [ids] }
};

export const wearItem = async (itemIdStr) => {
    const res = await api.post(`/items/wear/${itemIdStr}`);
    return res.data;
};

export const washItem = async (itemIdStr) => {
    const res = await api.post(`/items/wash/${itemIdStr}`);
    return res.data;
};

export const fetchWardrobeLayout = async () => {
    const res = await api.get('/wardrobe/layout');
    return res.data; // { total_items, sections: { A, B, C, D } }
};
