# -*- coding: utf-8 -*-
from django.conf.urls import patterns, url

urlpatterns = patterns('myproject.myapp.views',
    url(r'^show/$', 'show', name='show'),
)
