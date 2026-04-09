from flask import Blueprint, jsonify, request
import logging
from flask_jwt_extended import jwt_required, get_jwt_identity

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint for financial routes
financial_bp = Blueprint('financial', __name__, url_prefix='/api/financial')

@financial_bp.route('/overview', methods=['GET'])
@jwt_required()
def get_financial_overview():
    """Get financial overview for the user"""
    try:
        # Mock data - in a real app, this would come from the database
        mock_data = {
            'totalBalance': 1250000,
            'monthlyIncome': 450000,
            'monthlyExpenses': 320000,
            'savingsGoals': [
                {'name': 'New Mahindra Tractor', 'saved': 250000, 'target': 850000},
                {'name': 'Field Expansion Project', 'saved': 150000, 'target': 1200000},
            ],
            'recentTransactions': [
                {'type': 'income', 'description': 'Rice Sale', 'category': 'Crop Sales', 'amount': 150000, 'date': '2024-11-01'},
                {'type': 'expense', 'description': 'Fertilizer Purchase', 'category': 'Fertilizers', 'amount': 35000, 'date': '2024-10-30'},
                {'type': 'expense', 'description': 'Labor Costs', 'category': 'Labor', 'amount': 28000, 'date': '2024-10-28'},
            ],
            'upcomingPayments': []
        }
        return jsonify(mock_data), 200
    except Exception as e:
        logger.error(f"Error fetching financial overview: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/transactions', methods=['GET'])
@jwt_required()
def get_transactions():
    """Get user transactions"""
    try:
        limit = request.args.get('limit', 50, type=int)
        # Mock transactions
        mock_transactions = [
            {'id': 1, 'type': 'income', 'description': 'Rice Sale', 'category': 'Crop Sales', 'amount': 150000, 'date': '2024-11-01'},
            {'id': 2, 'type': 'expense', 'description': 'Fertilizer Purchase', 'category': 'Fertilizers', 'amount': 35000, 'date': '2024-10-30'},
            {'id': 3, 'type': 'expense', 'description': 'Labor Costs', 'category': 'Labor', 'amount': 28000, 'date': '2024-10-28'},
            {'id': 4, 'type': 'income', 'description': 'Wheat Sale', 'category': 'Crop Sales', 'amount': 120000, 'date': '2024-10-25'},
            {'id': 5, 'type': 'expense', 'description': 'Seed Purchase', 'category': 'Seeds', 'amount': 42000, 'date': '2024-10-20'},
        ]
        return jsonify(mock_transactions[:limit]), 200
    except Exception as e:
        logger.error(f"Error fetching transactions: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/budget-allocation', methods=['GET'])
@jwt_required()
def get_budget_allocation():
    """Get budget allocation data"""
    try:
        mock_allocation = {
            'seeds': 25,
            'fertilizers': 20,
            'labor': 30,
            'equipment': 15,
            'other': 10
        }
        return jsonify(mock_allocation), 200
    except Exception as e:
        logger.error(f"Error fetching budget allocation: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/recurring-transactions', methods=['GET'])
@jwt_required()
def get_recurring_transactions():
    try:
        mock_data = [
            {'id': 1, 'type': 'expense', 'description': 'Equipment Loan Payment', 'category': 'Equipment', 'amount': 12000, 'frequency': 'monthly', 'active': True},
            {'id': 2, 'type': 'expense', 'description': 'Insurance Premium', 'category': 'Insurance', 'amount': 8500, 'frequency': 'quarterly', 'active': True},
        ]
        return jsonify(mock_data), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/cash-flow', methods=['GET'])
@jwt_required()
def get_cash_flow():
    try:
        mock_data = {
            'projections': [],
            'currentSeason': 'fall',
            'seasons': {
                'spring': {'income': 200000, 'expenses': 450000},
                'summer': {'income': 150000, 'expenses': 250000},
                'fall': {'income': 850000, 'expenses': 350000},
                'winter': {'income': 100000, 'expenses': 150000}
            }
        }
        return jsonify(mock_data), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/loans', methods=['GET'])
@jwt_required()
def get_loans():
    try:
        mock_data = [
            {'id': 1, 'lender': 'Agricultural Bank of India', 'principal': 850000, 'interestRate': 4.25, 'termMonths': 84, 'remainingMonths': 67},
        ]
        return jsonify(mock_data), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/investments', methods=['GET'])
@jwt_required()
def get_investments():
    try:
        mock_data = [
            {'id': 1, 'name': 'Precision Planting System', 'type': 'technology', 'amount': 250000, 'expectedReturn': 12},
        ]
        return jsonify(mock_data), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/tax-data', methods=['GET'])
@jwt_required()
def get_tax_data():
    try:
        mock_data = {
            'currentYear': 2024,
            'categories': {'income': 1800000, 'deductibleExpenses': 1250000, 'depreciation': 250000, 'estimatedTax': 90000}
        }
        return jsonify(mock_data), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/savings-goals', methods=['GET'])
@jwt_required()
def get_savings_goals():
    try:
        mock_data = [
            {'id': 1, 'name': 'New Mahindra Tractor', 'targetAmount': 850000, 'currentAmount': 250000},
        ]
        return jsonify(mock_data), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/emergency-fund', methods=['GET'])
@jwt_required()
def get_emergency_fund():
    try:
        mock_data = {'currentAmount': 350000, 'targetAmount': 500000, 'monthlyExpenses': 85000}
        return jsonify(mock_data), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/seasonal-planning', methods=['GET'])
@jwt_required()
def get_seasonal_planning():
    try:
        mock_data = {'currentSavings': 450000, 'totalNeeded': 700000, 'projectedIncome': 1800000}
        return jsonify(mock_data), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@financial_bp.route('/expansion-plans', methods=['GET'])
@jwt_required()
def get_expansion_plans():
    try:
        mock_data = [{'id': 1, 'name': 'North Field Acquisition', 'estimatedCost': 1500000, 'timeframe': '2-3 years'}]
        return jsonify(mock_data), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
