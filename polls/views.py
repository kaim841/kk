from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return HttpResponse("hello world,you'are in the dniex")
# Create your vie
# ws here.
